"""Evaluate rules selected on one labeled dataset against a paired dataset.

Validation-result files must be the JSON text outputs produced by the paper's
Great Expectations runs. Rule definitions are matched across A/B results by
source run and expectation configuration; suite/window identifiers are ignored.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from .rule_selection import greedy_rule_selection_pairs

logger = logging.getLogger(__name__)
DEFAULT_SETTINGS = ("9_0", "10_0", "8_0", "10_2", "8_2", "9_2", "11_0", "11_2")


def _columns_from_error_label(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {
        column.strip()
        for group in value.split(";")
        for column in group.split("<->")
        if column.strip()
    }


def _dataset_metadata(
    path: Path,
) -> tuple[pd.DataFrame, set[int], set[tuple[int, str]]]:
    data = pd.read_csv(path)
    required = {"isDirty", "errCol"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    dirty_rows: set[int] = set()
    dirty_cells: set[tuple[int, str]] = set()
    for row_id, (is_dirty, err_col) in enumerate(zip(data["isDirty"], data["errCol"])):
        dirty_flag = (
            is_dirty.strip().lower() in {"1", "true", "yes"}
            if isinstance(is_dirty, str)
            else bool(is_dirty) if pd.notna(is_dirty) else False
        )
        if dirty_flag:
            dirty_rows.add(row_id)
            dirty_cells.update(
                (row_id, column)
                for column in _columns_from_error_label(err_col)
                if column in data.columns
            )
    return data, dirty_rows, dirty_cells
    
    
def _expectation_columns(expectation: dict[str, Any]) -> list[str]:
    kwargs = expectation.get("kwargs", {})
    columns: list[str] = []
    for key, value in kwargs.items():
        if "column" not in key.lower():
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, dict):
                columns.extend(str(column) for column in item)
            elif item is not None:
                columns.append(str(item))
    return list(dict.fromkeys(columns))


def _result_indices(result: dict[str, Any], data: pd.DataFrame) -> set[int]:
    result_data = result.get("result") or {}
    indices = result_data.get("unexpected_index_list")
    if indices is None:
        return set()
    # GX returns row indices for dataframe batches; reject indices outside this
    # dataset rather than accidentally interpreting value labels as row IDs.
    valid = set(data.index)
    indices = set()
    for index in result_data.get("unexpected_index_list", []):
        try:
            row_id = int(index)
        except (TypeError, ValueError):
            continue
        if row_id in data.index:
            indices.add(row_id)
    return indices


def _source_run(result_file: Path, dataset_stem: str) -> str:
    marker = f"BigName_{dataset_stem}_ValiBigOut_"
    if marker not in result_file.name:
        raise ValueError(f"Result file {result_file.name!r} lacks dataset marker {marker!r}")
    return result_file.name.split(marker, 1)[0]


def _load_rule_results(
    data_path: Path,
    results_directory: Path,
    patterns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data, dirty_rows, dirty_cells = _dataset_metadata(data_path)
    marker = f"BigName_{data_path.stem}_ValiBigOut_"
    files: set[Path] = set()
    for pattern in patterns:
        files.update(results_directory.glob(pattern))
    files = {
        path for path in files
        if marker in path.name and re.search(r"_Size\d+\.txt$", path.name)
    }
    if not files:
        raise FileNotFoundError(
            f"No validation result files for {data_path.stem!r} in {results_directory}; "
            f"patterns={patterns!r}"
        )

    rules: dict[str, dict[str, Any]] = {}
    for result_file in sorted(files):
        with result_file.open(encoding="utf-8") as handle:
            encoded = json.load(handle)
        if isinstance(encoded, str):
            try:
                validation = json.loads(encoded)
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON validation output %s", result_file)
                continue
        elif isinstance(encoded, dict):
            validation = encoded
        else:
            continue
        if not isinstance(validation, dict):
            continue
        source_run = _source_run(result_file, data_path.stem)

        for result in validation.get("results", []):
            exception = result.get("exception_info", {})
            if exception.get("raised_exception", False):
                continue
            expectation = result.get("expectation_config", {})
            columns = [column for column in _expectation_columns(expectation) if column in data.columns]
            if not columns:
                continue

            identity = json.dumps(
                [source_run, expectation],
                sort_keys=True,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
            predicted_rows = _result_indices(result, data)
            target_rows = {
                row_id for row_id, column in dirty_cells if column in columns
            }
            # The source script credits null-valued target cells as detected
            # positives for rules evaluated on those cells.
            missing_target_rows = {
                row_id
                for column in columns
                for row_id in dirty_rows
                if (row_id, column) in dirty_cells and pd.isna(data.at[row_id, column])
            }
            predicted_rows.update(missing_target_rows & target_rows)
            record = rules.setdefault(identity, {
                "ri": identity,
                "columns": columns,
                "predicted_rows": set(),
                "target_rows": target_rows,
                "source_run": source_run,
            })
            record["predicted_rows"].update(predicted_rows)

    if not rules:
        raise ValueError(f"No usable rules found for {data_path}")

    rule_rows = []
    pair_rows = []
    for identity, record in rules.items():
        true_positive = len(record["predicted_rows"] & record["target_rows"])
        false_positive = len(record["predicted_rows"] - record["target_rows"])
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        rule_rows.append({
            "ri": identity,
            "col_set": ";".join(record["columns"]),
            "precision": precision,
        })
        for row_id in record["predicted_rows"]:
            matching_columns = [
                column for column in record["columns"]
                if (row_id, column) in dirty_cells
            ]
            pair_rows.append({
                "ri": identity,
                "pk": row_id,
                "col": ";".join(matching_columns or record["columns"]),
                "isDirty": int(bool(matching_columns)),
            })

    return (
        pd.DataFrame(rule_rows, columns=["ri", "col_set", "precision"]),
        pd.DataFrame(pair_rows, columns=["ri", "pk", "col", "isDirty"]),
    )


def _pattern(pattern_name: str, dataset_stem: str, setting: str) -> str:
    if pattern_name == "pci":
        run_prefix = "new_OnlyRelevanRules"
    elif pattern_name in {"hospital", "flights", "beers"}:
        run_prefix = f"new_{pattern_name}_OnlyRelevanRules"
    elif pattern_name in {"tax", "tax_gemma4", "tax_glm"}:
        run_prefix = f"new_{pattern_name}*_OnlyRelevanRules"
    else:
        raise ValueError(f"Unsupported dataset type: {pattern_name}")

    return (
        f"resDf_all__*{setting}_*_{run_prefix}_valChange_newValInfo_blub0_"
        f"*BigName_{dataset_stem}_ValiBigOut_*_Size*.txt"
    )


def _load_dataset_path(value: str, data_directory: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = data_directory / path
    return path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")


def _evaluate_setting(
    pattern_name: str,
    setting: str,
    reduction_path: Path,
    coverage_path: Path,
    results_directory: Path,
) -> list[dict[str, Any]]:
    reduction_rules, reduction_pairs = _load_rule_results(
        reduction_path,
        results_directory,
        [_pattern(pattern_name, reduction_path.stem, setting)],
    )
    coverage_rules, coverage_pairs = _load_rule_results(
        coverage_path,
        results_directory,
        [_pattern(pattern_name, coverage_path.stem, setting)],
    )
    shared = set(reduction_rules["ri"]) & set(coverage_rules["ri"])
    if not shared:
        raise ValueError(
            f"No matching rule definitions for setting {setting!r} "
            f"between {reduction_path.stem!r} and {coverage_path.stem!r}"
        )

    identity_to_id = {identity: index for index, identity in enumerate(sorted(shared))}
    for frame in (reduction_rules, coverage_rules, reduction_pairs, coverage_pairs):
        frame.drop(frame.loc[~frame["ri"].isin(shared)].index, inplace=True)
        frame["ri"] = frame["ri"].map(identity_to_id)

    output_rows = []
    for threshold_percent in range(0, 105, 5):
        threshold = threshold_percent / 100
        selected, _, _ = greedy_rule_selection_pairs(
            reduction_pairs, reduction_rules, precision_required=threshold
        )
        selected_set = set(selected)
        selected_dirty = coverage_pairs[
            coverage_pairs["ri"].isin(selected_set) & (coverage_pairs["isDirty"] == 1)
        ]
        covered_pairs: set[tuple[int, str]] = set()
        coverage_by_rule = {}
        for row in selected_dirty.itertuples(index=False):
            pairs = {
                (row.pk, column.strip())
                for column in str(row.col).split(";")
                if column.strip()
            }
            coverage_by_rule.setdefault(row.ri, set()).update(pairs - covered_pairs)
            covered_pairs.update(pairs)

        selected_clean = coverage_pairs[
            coverage_pairs["ri"].isin(selected_set) & (coverage_pairs["isDirty"] == 0)
        ]
        clean_rule_columns = coverage_rules.set_index("ri")["col_set"].to_dict()
        clean_pairs = {
            (row.pk, clean_rule_columns[row.ri])
            for row in selected_clean.itertuples(index=False)
        }
        output_rows.append({
            "setting": setting,
            "precValReq": round(threshold_percent / 100, 2),
            "total_pairs": len(covered_pairs),
            "clean_val": len(clean_pairs),
            "covered_pairs": covered_pairs,
            "selected_rules": selected,
            "coverage_by_rule": coverage_by_rule,
        })
    return output_rows


def run_paired_coverage(
    *,
    pattern_name: str,
    data_directory: str | Path,
    results_directory: str | Path,
    output_directory: str | Path,
    reduction_file: str,
    coverage_file: str,
    settings: tuple[str, ...] = DEFAULT_SETTINGS,
) -> list[Path]:
    """Select rules on A and report dirty/clean cell coverage on B by precision threshold."""
    data_directory = Path(data_directory)
    results_directory = Path(results_directory)
    output_directory = Path(output_directory)
    reduction_path = _load_dataset_path(reduction_file, data_directory)
    coverage_path = _load_dataset_path(coverage_file, data_directory)
    for path in (reduction_path, coverage_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for setting in settings:
        try:
            rows = _evaluate_setting(
                pattern_name, setting, reduction_path, coverage_path, results_directory
            )
        except FileNotFoundError as error:
            logger.warning("Skipping setting %s: %s", setting, error)
            continue
        output_path = output_directory / (
            f"paired_coverage_{reduction_path.stem}_to_{coverage_path.stem}_"
            f"{pattern_name}_{setting}.csv"
        )
        pd.DataFrame(rows).to_csv(output_path, index=False)
        outputs.append(output_path)
    if not outputs:
        raise FileNotFoundError(
            "No paired results were produced. Check the dataset names, setting suffixes, "
            "and that validation-result files exist for both datasets."
        )
    return outputs




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select rules on a labeled A dataset and evaluate their coverage on B"
    )
    parser.add_argument("--pattern-name", required=True, choices=("hospital", "flights", "beers", "tax", "tax_gemma4", "tax_glm", "pci"))
    parser.add_argument("--data-in", type=Path, required=True, help="Directory containing the A/B CSV files")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing GX validation-result text files")
    parser.add_argument("--data-out", type=Path, required=True)
    parser.add_argument("--reduction-data-file", required=True, help="A dataset filename or path, with or without .csv")
    parser.add_argument("--coverage-data-file", required=True, help="B dataset filename or path, with or without .csv")
    parser.add_argument(
        "--settings", nargs="+", default=list(DEFAULT_SETTINGS),
        help="Run/prompt/expectation setting suffixes to include, e.g. 9_0 9_2",
    )
    arguments = parser.parse_args()
    outputs = run_paired_coverage(
        pattern_name=arguments.pattern_name,
        data_directory=arguments.data_in,
        results_directory=arguments.results_dir,
        output_directory=arguments.data_out,
        reduction_file=arguments.reduction_data_file,
        coverage_file=arguments.coverage_data_file,
        settings=tuple(arguments.settings),
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
