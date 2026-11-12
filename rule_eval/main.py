"""Command-line entry point for the LeDQeR rule-evaluation pipeline.

Run from the repository root:

    python -m rule_eval.main --config rule_eval/parameter.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd

from .evaluate_rules import (
    check_all_rules,
    get_suite_from_json,
    save_suite_as_json,
)
from .parallel_eval import validate_large_dataset
from .prepare_selection_data import prepare_selection_data_from_suite_li

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load an evaluation configuration file."""
    with Path(config_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the compact and per-dataset configuration formats."""
    if config.get("dataSets"):
        specs = []
        for raw_spec in config["dataSets"]:
            spec = dict(raw_spec)
            spec.setdefault("name", spec.get("generationResult"))
            spec.setdefault("evaluationData", [])
            if isinstance(spec["evaluationData"], str):
                spec["evaluationData"] = [spec["evaluationData"]]
            specs.append(spec)
        return specs

    generation_files = config.get("dataFiles", [])
    evaluation_files = config.get("dataFilesBig", [])
    if len(generation_files) == len(evaluation_files):
        evaluation_by_generation = [[name] for name in evaluation_files]
    else:
        evaluation_by_generation = [list(evaluation_files) for _ in generation_files]
    return [
        {
            "name": generation_file,
            "generationResult": generation_file,
            "evaluationData": evaluation_by_generation[index],
            "allColumnCount": config.get("allColumnCount"),
        }
        for index, generation_file in enumerate(generation_files)
    ]


def _require_column_count(spec: dict[str, Any]) -> int:
    count = spec.get("allColumnCount")
    if not isinstance(count, int) or count < 1:
        raise ValueError(
            f"Dataset {spec.get('name')!r} needs a positive allColumnCount"
        )
    return count


def _result_stem(config: dict[str, Any], spec: dict[str, Any]) -> str:
    stem = str(spec["name"])
    if config.get("useOnlyRelevantRules", True):
        stem += "_onlyRelevantRules"
    return stem


def _read_csv(directory: Path, name: str) -> pd.DataFrame:
    path = directory / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def run_comprehensive_evaluation(
    config: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[list[dict], gx.ExpectationSuite, gx.ExpectationSuite, pd.DataFrame]:
    """Parse and evaluate generated rules for one dataset."""
    source = _read_csv(Path(config["dataIn"]), spec["generationResult"])
    if "isDirty" in source.columns:
        source = source.query("isDirty == 1").reset_index(drop=True)

    suite_list, suite_all, suite_relevant = check_all_rules(
        source,
        all_col_count=_require_column_count(spec),
        use_only_relevant_rules=config.get("useOnlyRelevantRules", True),
    )
    return suite_list, suite_all, suite_relevant, source


def save_results(
    config: dict[str, Any],
    stem: str,
    suite_list: list[dict],
    source: pd.DataFrame,
    suite_all: gx.ExpectationSuite,
    suite_relevant: gx.ExpectationSuite,
) -> None:
    """Persist evaluation rows, selection inputs, and both rule suites."""
    data_out = Path(config["dataOut"])
    suite_out = Path(config["suiteOut"])
    data_out.mkdir(parents=True, exist_ok=True)
    suite_out.mkdir(parents=True, exist_ok=True)

    with (data_out / f"{stem}.pkl").open("wb") as handle:
        pickle.dump(suite_list, handle)
    pd.DataFrame(suite_list).to_csv(data_out / f"{stem}.csv", index=False)

    pairs, rules = prepare_selection_data_from_suite_li(
        suite_list,
        source,
        use_only_relevant_rules=config.get("useOnlyRelevantRules", True),
    )
    pairs.to_csv(data_out / f"{stem}_df_pairs.csv", index=False)
    rules.to_csv(data_out / f"{stem}_df_rules.csv", index=False)
    save_suite_as_json(suite_all, str(suite_out / f"{stem}_suite_all.json"))
    save_suite_as_json(
        suite_relevant,
        str(suite_out / f"{stem}_suite_relevant.json"),
    )


def run_large_evaluation(
    config: dict[str, Any],
    spec: dict[str, Any],
    stem: str,
    suite: gx.ExpectationSuite,
) -> list[dict[str, Any]]:
    """Evaluate a selected suite on every configured held-out dataset."""
    data_in = Path(config["dataIn"])
    data_out = Path(config["dataOut"])
    data_out.mkdir(parents=True, exist_ok=True)
    summaries = []

    for evaluation_name in spec.get("evaluationData", []):
        data = _read_csv(data_in, evaluation_name)
        result = validate_large_dataset(
            data,
            suite,
            window_size=config.get("windowSize", 100_000),
            suite_window_size=config.get("suiteWindowSize", 1),
            max_workers=config.get("maxWorkers", 1),
            regex_sample_rows=config.get("regexPrecheckRows", 200),
            regex_time_budget=config.get("regexPrecheckSeconds", 0.01),
        )
        summary = {
            "dataset": spec["name"],
            "evaluationData": evaluation_name,
            "totalRows": result["total_rows"],
            "windowCount": result["window_count"],
            "rulesRun": result["suite_stats"]["rule_count"],
            "rulesNotRun": result["suite_stats"]["not_run_count"],
            "totalExecutionTime": result["suite_stats"]["total_time"],
            "removedRegexRules": len(
                result["suite_stats"]["removed_regex_rules"]
            ),
        }
        summaries.append(summary)
        with (data_out / f"{stem}_{evaluation_name}_large.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(result, handle, indent=2, default=str)

    if summaries:
        pd.DataFrame(summaries).to_csv(
            data_out / f"{stem}_large_summary.csv", index=False
        )
    return summaries


def run(config_path: str | Path) -> list[dict[str, Any]]:
    """Run the configured evaluation pipeline and return large-test summaries."""
    config = load_config(config_path)
    specs = _dataset_specs(config)
    if not specs:
        raise ValueError("No datasets configured")

    all_summaries = []
    for spec in specs:
        stem = _result_stem(config, spec)
        suite_kind = (
            "relevant" if config.get("useOnlyRelevantRules", True) else "all"
        )
        suite_path = Path(config["suiteOut"]) / f"{stem}_suite_{suite_kind}.json"

        if config.get("onlyRunBig", False):
            if not suite_path.is_file():
                raise FileNotFoundError(suite_path)
            evaluation_suite = get_suite_from_json(str(suite_path))
        else:
            suite_list, suite_all, suite_relevant, source = (
                run_comprehensive_evaluation(config, spec)
            )
            save_results(
                config,
                stem,
                suite_list,
                source,
                suite_all,
                suite_relevant,
            )
            evaluation_suite = (
                suite_relevant
                if config.get("useOnlyRelevantRules", True)
                else suite_all
            )
            logger.info(
                "%s: evaluated %s generated-response rows",
                spec["name"],
                len(suite_list),
            )

        if config.get("isRunBigTest", False) or config.get("onlyRunBig", False):
            all_summaries.extend(
                run_large_evaluation(config, spec, stem, evaluation_suite)
            )
    return all_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated DQ rules")
    parser.add_argument(
        "--config",
        default="rule_eval/parameter.json",
        help="Evaluation configuration JSON",
    )
    arguments = parser.parse_args()
    run(arguments.config)


if __name__ == "__main__":
    main()
