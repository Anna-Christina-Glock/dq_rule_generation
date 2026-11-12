"""Utilities for parsing, evaluating, and reducing LeDQeR-generated rules."""

from .evaluate_rules import (
    check_all_rules,
    extract_expectations_from_json,
    validate_rule_on_dataframe,
    validate_with_typed_between,
)
from .parallel_eval import validate_large_dataset
from .prepare_selection_data import (
    prepare_selection_data_from_file,
    prepare_selection_data_from_suite_li,
)

from .rule_selection import (
    get_coverage_for_precision_levels,
    greedy_rule_selection,
    greedy_rule_selection_pairs,
)

__all__ = [
    "check_all_rules",
    "extract_expectations_from_json",
    "get_coverage_for_precision_levels",
    "greedy_rule_selection",
    "greedy_rule_selection_pairs",
    "validate_large_dataset",
    "validate_rule_on_dataframe",
    "validate_with_typed_between",
    "prepare_selection_data_from_file",
    "prepare_selection_data_from_suite_li",
]

__version__ = "1.0.0"
