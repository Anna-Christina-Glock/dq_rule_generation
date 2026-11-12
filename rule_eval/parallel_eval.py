"""Windowed evaluation for rule suites on large dataframes.

Great Expectations keeps dataframe batches on a shared context. Validation is
therefore deliberately serial even though the historical module name is kept
for compatibility with existing experiment scripts.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import great_expectations as gx
import pandas as pd

from .evaluate_rules import validate_with_typed_between

logger = logging.getLogger(__name__)


def create_suite_parts(
    suite_all: gx.ExpectationSuite,
    suite_window_size: int,
) -> list[gx.ExpectationSuite]:
    """Split a suite into deterministic, ordered partitions."""
    if suite_window_size < 1:
        raise ValueError("suite_window_size must be at least 1")

    expectations = list(suite_all.expectations)
    parts = []
    for start in range(0, len(expectations), suite_window_size):
        part = gx.ExpectationSuite(name=f"suite_part_{start}")
        for expectation in expectations[start : start + suite_window_size]:
            part.add_expectation(expectation.copy())
        parts.append(part)
    return parts


def precheck_suites_regex(
    suites: list[gx.ExpectationSuite],
    df_data: pd.DataFrame,
    sample_rows: int = 200,
    time_budget: float = 0.01,
) -> tuple[list[gx.ExpectationSuite], list[dict[str, Any]]]:
    """Remove invalid or unusually slow regex rules before full-data validation.

    The check preserves all non-regex expectations and records every removed
    expectation. It does not rewrite model-generated regular expressions.
    """
    if df_data.empty:
        return suites, []

    checked = []
    removed = []
    regex_types = ("to_match_regex", "to_not_match_regex")
    for suite_index, suite in enumerate(suites):
        kept = []
        for expectation in suite.expectations:
            expectation_type = expectation.expectation_type
            if not any(token in expectation_type for token in regex_types):
                kept.append(expectation)
                continue

            column = getattr(expectation, "column", None)
            pattern = getattr(expectation, "regex", None)
            if not column or not pattern or column not in df_data.columns:
                kept.append(expectation)
                continue

            started = time.perf_counter()
            try:
                compiled = re.compile(pattern)
                values = df_data[column].dropna().astype(str).head(sample_rows)
                matched_rows = sum(bool(compiled.search(value)) for value in values)
            except re.error as error:
                removed.append(
                    {
                        "suite_id": suite_index,
                        "column": column,
                        "expectation_type": expectation_type,
                        "regex": pattern,
                        "reason": "invalid_regex",
                        "error": str(error),
                    }
                )
                continue

            elapsed = time.perf_counter() - started
            if elapsed > time_budget:
                removed.append(
                    {
                        "suite_id": suite_index,
                        "column": column,
                        "expectation_type": expectation_type,
                        "regex": pattern,
                        "reason": "slow_regex_precheck",
                        "sample_rows": len(values),
                        "matched_rows": matched_rows,
                        "elapsed_s": elapsed,
                    }
                )
                continue
            kept.append(expectation)

        if kept:
            checked_suite = gx.ExpectationSuite(name=suite.name)
            for expectation in kept:
                checked_suite.add_expectation(expectation.copy())
            checked.append(checked_suite)
    return checked, removed


def validate_single_suite(
    suite: gx.ExpectationSuite,
    df: pd.DataFrame,
    result_format: dict,
    window_id: int,
    suite_id: int,
) -> dict[str, Any]:
    """Validate one suite partition and return timing/error metadata."""
    started = time.perf_counter()
    try:
        result = validate_with_typed_between(df, suite, result_format)
        return {
            "window_id": window_id,
            "suite_id": suite_id,
            "result": result,
            "error": None,
            "execution_time": time.perf_counter() - started,
        }
    except Exception as error:  # preserve the remaining suite partitions
        logger.exception("Suite %s failed in window %s", suite_id, window_id)
        return {
            "window_id": window_id,
            "suite_id": suite_id,
            "result": None,
            "error": str(error),
            "execution_time": time.perf_counter() - started,
        }


def validate_batch_parallel(
    df: pd.DataFrame,
    suite: gx.ExpectationSuite,
    suite_window_size: int = 1,
    max_workers: int = 1,
    result_format: dict | None = None,
    regex_sample_rows: int = 200,
    regex_time_budget: float = 0.01,
    window_id: int = 0,
    precheck_regex: bool = True,
) -> dict[str, Any]:
    """Validate every suite partition on one dataframe window.

    ``max_workers`` remains in the public interface for configuration
    compatibility. Values above one are clamped because GX dataframe batches
    share an ephemeral context and concurrent validation can use stale batches.
    """
    if max_workers != 1:
        logger.warning("Clamping max_workers=%s to 1 for GX context safety", max_workers)

    if result_format is None:
        result_format = {
            "result_format": "COMPLETE",
            "include_unexpected_rows": True,
        }

    suites = create_suite_parts(suite, suite_window_size)
    if precheck_regex:
        suites, removed_rules = precheck_suites_regex(
            suites,
            df,
            sample_rows=regex_sample_rows,
            time_budget=regex_time_budget,
        )
    else:
        removed_rules = []
    results = [
        validate_single_suite(part, df, result_format, window_id, suite_id)
        for suite_id, part in enumerate(suites)
    ]
    successful = sum(result["error"] is None for result in results)
    total_time = sum(result["execution_time"] for result in results)
    return {
        "results": results,
        "success_count": successful,
        "error_count": len(results) - successful,
        "total_execution_time": total_time,
        "average_execution_time": total_time / len(results) if results else 0.0,
        "num_suites": len(results),
        "removed_regex_rules": removed_rules,
    }


def validate_large_dataset(
    df: pd.DataFrame,
    suite: gx.ExpectationSuite,
    window_size: int = 100_000,
    suite_window_size: int = 1,
    max_workers: int = 1,
    result_format: dict | None = None,
    regex_sample_rows: int = 200,
    regex_time_budget: float = 0.01,
) -> dict[str, Any]:
    """Validate a large dataframe in row windows and aggregate run metadata."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    checked_parts, removed_rules = precheck_suites_regex(
        create_suite_parts(suite, suite_window_size),
        df,
        sample_rows=regex_sample_rows,
        time_budget=regex_time_budget,
    )
    checked_suite = gx.ExpectationSuite(name=f"{suite.name}_regex_checked")
    for part in checked_parts:
        for expectation in part.expectations:
            checked_suite.add_expectation(expectation.copy())

    window_stats = []
    suite_stats = {
        "success_count": 0,
        "error_count": 0,
        "total_time": 0.0,
        "rule_count": 0,
        "not_run_count": 0,
        "removed_regex_rules": removed_rules,
    }

    for window_id, start in enumerate(range(0, len(df), window_size)):
        end = min(start + window_size, len(df))
        window = df.iloc[start:end]
        started = time.perf_counter()
        window_result = validate_batch_parallel(
            window,
            checked_suite,
            suite_window_size=suite_window_size,
            max_workers=max_workers,
            result_format=result_format,
            window_id=window_id,
            precheck_regex=False,
        )

        rule_count = 0
        not_run_count = 0
        for suite_result in window_result["results"]:
            validation = suite_result["result"]
            if validation is None:
                continue
            for result in validation.get("results", []):
                if result.get("result"):
                    rule_count += 1
                else:
                    not_run_count += 1

        window_stats.append(
            {
                "window_start": start,
                "window_end": end,
                "row_count": len(window),
                "execution_time": time.perf_counter() - started,
                "suite_success": window_result["success_count"],
                "suite_errors": window_result["error_count"],
                "rule_count": rule_count,
                "not_run_count": not_run_count,
            }
        )
        suite_stats["success_count"] += window_result["success_count"]
        suite_stats["error_count"] += window_result["error_count"]
        suite_stats["total_time"] += window_result["total_execution_time"]
        suite_stats["rule_count"] += rule_count
        suite_stats["not_run_count"] += not_run_count

    return {
        "window_stats": window_stats,
        "suite_stats": suite_stats,
        "total_rows": len(df),
        "window_count": len(window_stats),
    }
