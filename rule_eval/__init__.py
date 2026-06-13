"""
RuleEval Package
LLM-based Data Quality Rule Evaluation

This package provides tools for evaluating LLM-generated data quality rules using Great Expectations.
"""

from .evaluate_rules import (
    # Core validation functions
    get_batch,
    calculate_precision_recall,
    get_validation_result,
    get_precision_recall_per_row,
    get_precision_recall_per_rule_col,
    get_precision_recall_per_col,
    fill_expectation_suite,
    extract_expectations_from_json,
    validate_rule_on_dataframe,
    # Data type correction
    to_type,
    to_type_df_col,
    get_dtype_corr_df,
    # Suite persistence
    save_suite_as_json,
    get_suite_from_json,
    # Comprehensive evaluation
    check_all_rules,
)

from .parallel_eval import (
    # Parallel execution functions
    create_suite_parts,
    validate_single_suite,
    validate_batch_parallel,
    validate_large_dataset,
)

from .rule_selection import (
    # Rule selection algorithms
    greedy_rule_selection_pairs,
    greedy_rule_selection,
    get_coverage_for_precision_levels,
)

from .prepare_selection_data import (
    # Prepare selection data
    prepare_selection_data_from_suite_li,
    prepare_selection_data_from_file,
)

from .plot_results import (
    # Plotting and result visualization
    getF1,
    load_latest_per_unique_file,
    load_and_process_data,
    transform_data_for_plotting,
    create_model_comparison_plot,
    create_coverage_plot_models,
    export_execution_stats_csv,
)

# Main entry point
from .main import main as run_evaluation

__all__ = [
    # Core validation
    'get_batch', 'calculate_precision_recall', 'get_validation_result',
    'get_precision_recall_per_row', 'get_precision_recall_per_rule_col',
    'get_precision_recall_per_col', 'fill_expectation_suite',
    'extract_expectations_from_json', 'validate_rule_on_dataframe',
    # Data type correction
    'to_type', 'to_type_df_col', 'get_dtype_corr_df',
    # Suite persistence
    'save_suite_as_json', 'get_suite_from_json',
    # Comprehensive evaluation
    'check_all_rules',
    # Parallel execution
    'create_suite_parts', 'validate_single_suite',
    'validate_batch_parallel', 'validate_large_dataset',
    # Rule selection
    'greedy_rule_selection_pairs', 'greedy_rule_selection',
    'get_coverage_for_precision_levels',
    # Prepare selection data
    'prepare_selection_data_from_suite_li', 'prepare_selection_data_from_file',
    # Plotting
    'getF1', 'load_latest_per_unique_file', 'load_and_process_data',
    'transform_data_for_plotting', 'create_model_comparison_plot',
    'create_coverage_plot_models', 'export_execution_stats_csv',
    # Main entry point
    'run_evaluation',
]

__version__ = "1.0.0"
