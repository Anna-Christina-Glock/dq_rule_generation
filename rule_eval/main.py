"""
Main Runner for Rule Evaluation Pipeline

This script provides a one-click interface for running the complete rule evaluation
pipeline. It reads a configuration file (parameter.json) and orchestrates the
evaluation, including comprehensive rule validation and parallel large-dataset testing.

Usage:
    python pub/dq_rule_generation/rule_eval/main.py --config path/to/parameter.json
"""

import argparse
import json
import time
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

import great_expectations as gx
import pandas as pd

from parallel_eval import validate_large_dataset
from evaluate_rules import check_all_rules
from prepare_selection_data import prepare_selection_data_from_suite_li
from evaluate_rules import (
    save_suite_as_json,
    get_suite_from_json,
    get_precision_recall_from_validation,
    get_precision_recall_from_validation_per_rule_col,
    get_precision_recall_from_validation_per_col,
    get_dtype_corr_df,
    _get_default_precision_recall,
    _get_default_precision_recall_per_rule_col,
    _get_default_precision_recall_per_col,
    get_validation_result
)


def load_config(config_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_comprehensive_evaluation(
    config: dict,
    poll_data_name: str,
    use_data_type_field: bool,
    use_only_relevant_rules: bool
) -> tuple:
    """
    Run comprehensive rule evaluation for a single dataset.
    
    Args:
        config: Configuration dictionary
        poll_data_name: Name of the polluted data file
        use_data_type_field: Whether to apply data type corrections
        use_only_relevant_rules: Whether to use only relevant rules
        
    Returns:
        Tuple of (suite_list, suite_all, suite_relevant, file_suffix)
    """
    directory = Path(config['dataIn'])
    df_res_all = pd.read_csv(directory / f'{poll_data_name}.csv')
    
    # Filter for dirty rows only (as per original behavior)
    df_res_all = df_res_all.query('isDirty')
    
    # Build file suffix
    str_add = ''
    if use_data_type_field:
        str_add += '_convertData'
    if use_only_relevant_rules:
        str_add += '_OnlyRelevanRules'
    fname = f'{poll_data_name}{str_add}_valChange_newValInfo_blub0'
    
    # Run evaluation
    suite_li, suite_all, suite_relevant = check_all_rules(
        df_res_all,
        all_col_count=14,
        use_data_type_field=use_data_type_field,
        use_only_relevant_rules=use_only_relevant_rules
    )
    
    return suite_li, suite_all, suite_relevant, fname


def save_results(
    config: dict,
    suite_li: list,
    fname: str,
    suite_all: gx.ExpectationSuite = None,
    suite_relevant: gx.ExpectationSuite = None
) -> None:
    """Save evaluation results to files."""
    data_out = Path(config['dataOut'])
    suite_out = Path(config['suiteOut'])
    
    # Save as pickle
    with open(data_out / f'{fname}.pkl', 'wb') as f:
        pickle.dump(suite_li, f)
    
    # Save as CSV
    try:
        pd.DataFrame(suite_li).to_csv(data_out / f'{fname}.csv')
    except Exception as e:
        print(f"CSV save error: {e}")
    
    # Prepare and save selection data
    try:
        print("Preparing selection data (df_pairs, df_rules)...")
        # Re-read the full dataset to get pk values
        df_res_all_full = pd.read_csv(directory / f'{poll_data_name}.csv')
        # Use isDirty=1 only (dirty rows)
        df_res_all_dirty = df_res_all_full.query('isDirty == 1').reset_index(drop=True)
        
        df_pairs, df_rules = prepare_selection_data_from_suite_li(suite_li, df_res_all_dirty, use_only_relevant_rules=use_only_relevant_rules)
        
        # Save selection data
        df_pairs.to_csv(data_out / f'{fname}_df_pairs.csv', index=False)
        df_rules.to_csv(data_out / f'{fname}_df_rules.csv', index=False)
        print(f"Selection data saved: {fname}_df_pairs.csv, {fname}_df_rules.csv")
    except Exception as e:
        print(f"Selection data save error: {e}")
    
    # Save suites
    if suite_all is not None:
        try:
            save_suite_as_json(suite_all, suite_out / f'{fname}_suite_all.json')
        except Exception as e:
            print(f"Suite save error: {e}")
            
    if suite_relevant is not None:
        try:
            save_suite_as_json(suite_relevant, suite_out / f'{fname}_suite_relevant.json')
        except Exception as e:
            print(f"Relevant suite save error: {e}")


def print_evaluation_stats(suite_li: list) -> None:
    """Print evaluation statistics."""
    rules_not_run = 0
    rules_run = 0
    
    for idx, row in pd.DataFrame(suite_li).loc[:, ['rule_count', 'exception_count']].iterrows():
        if row['rule_count'] == row['exception_count']:
            rules_not_run += 1
        else:
            rules_run += 1
    
    print(f"\nRules Not Run: {rules_not_run}")
    print(f"Rules Run: {rules_run}")
    print(f"Total Rules: {len(suite_li)}")


def run_big_test(
    config: dict,
    poll_data_big_name: str,
    suite_all: gx.ExpectationSuite,
    fname: str
) -> dict:
    """
    Run parallel validation on a large dataset.
    
    Args:
        config: Configuration dictionary
        poll_data_big_name: Name of the big data file
        suite_all: The suite to validate against
        fname: Base filename for output
        
    Returns:
        Dictionary with validation statistics
    """
    directory = Path(config['dataIn'])
    data_out = Path(config['dataOut'])
    
    df_data_big = pd.read_csv(directory / f'{poll_data_big_name}.csv')
    
    # Extract config parameters
    max_rows = 1000000  # Default, can be overridden
    max_workers = config.get('max_workers', 10)
    window_size = config.get('window_size', 1000)
    suite_window_size = config.get('suite_window_size', 10)
    
    # Run parallel validation
    result = validate_large_dataset(
        df=df_data_big,
        suite=suite_all,
        window_size=window_size,
        suite_window_size=suite_window_size,
        max_workers=max_workers
    )
    
    # Extract statistics
    run_rule = result['suite_stats']['rule_count']
    not_run_rule = result['suite_stats']['not_run_count']
    
    stats = {
        'pollDataBigName': poll_data_big_name,
        'total_rows': result['total_rows'],
        'window_count': result['window_count'],
        'run_rule': run_rule,
        'not_run_rule': not_run_rule,
        'rule_all': run_rule + not_run_rule,
        'total_execution_time': result['suite_stats']['total_time'],
        'avg_suite_time': result['suite_stats']['total_time'] / len(result['window_stats']) if result['window_stats'] else 0
    }
    
    # Save statistics
    pd.DataFrame([stats]).to_csv(
        data_out / f'{fname}_BigName_{poll_data_big_name}_ExcecPara.csv',
        index=False
    )
    
    # Save per-window timing
    window_stats_df = pd.DataFrame(result['window_stats'])
    window_stats_df.to_csv(
        data_out / f'{fname}_BigName_{poll_data_big_name}_resTimeLi.csv',
        index=False
    )
    
    return stats


def main(config_path: str = None) -> None:
    """
    Main entry point for the rule evaluation pipeline.
    
    Args:
        config_path: Path to the configuration file. If None, uses default location.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run rule evaluation pipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to parameter.json configuration file')
    parser.add_argument('--config-dir', type=str, default='rule_eval',
                        help='Directory containing parameter.json (default: llmTests/rule_Generation/greatExpectations/config)')
    args = parser.parse_args()
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = Path(args.config_dir) / 'parameter.json'
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(str(config_path))
    
    # Extract common config values
    use_data_type_field = config.get('useDataTypeField', False)
    use_only_relevant_rules = config.get('useOnlyRelevantRules', True)
    is_run_big_test = config.get('isRunBigTest', False)
    only_run_big = config.get('onlyRunBig', False)
    
    # Get data files
    poll_data_name_vec = config.get('dataFiles', [])
    poll_data_big_name_vec = config.get('dataFilesBig', [])
    
    # Store all big test results
    run_big_li = []
    
    # Phase 1: Comprehensive evaluation for all datasets
    if not only_run_big:
        print("\n" + "=" * 60)
        print("Phase 1: Comprehensive Rule Evaluation")
        print("=" * 60)
        
        start_time_extract = time.time()
        
        for poll_data_name in poll_data_name_vec:
            print(f"\nProcessing: {poll_data_name}")
            
            suite_li, suite_all, suite_relevant, fname = run_comprehensive_evaluation(
                config, poll_data_name, use_data_type_field, use_only_relevant_rules
            )
            
            # Print statistics
            print(f"Total rules processed: {len(suite_li)}")
            print_evaluation_stats(suite_li)
            
            # Save results
            save_results(config, suite_li, fname, suite_all, suite_relevant)
            print(f"Results saved with filename: {fname}")
        
        end_time_extract = time.time()
        print(f"\nComprehensive evaluation completed in: {end_time_extract - start_time_extract:.2f} seconds")
    
    # Phase 2: Parallel validation on large datasets
    if is_run_big_test or only_run_big:
        print("\n" + "=" * 60)
        print("Phase 2: Parallel Large Dataset Validation")
        print("=" * 60)
        
        for poll_data_big_name in poll_data_big_name_vec:
            print(f"\nProcessing Big Dataset: {poll_data_big_name}")
            
            if only_run_big:
                # Load existing suite for big test
                fname = poll_data_big_name  # Will be overridden
                # Find the relevant suite for this big dataset
                for poll_data_name in poll_data_name_vec:
                    str_add = ''
                    if use_data_type_field:
                        str_add += '_convertData'
                    if use_only_relevant_rules:
                        str_add += '_OnlyRelevanRules'
                    fname = f'{poll_data_name}{str_add}_valChange_newValInfo_blub0'
                    suite_path = Path(config['suiteOut']) / f'{fname}_suite_relevant.json'
                    if suite_path.exists():
                        print(f"Loading suite from: {suite_path}")
                        suite_all = get_suite_from_json(str(suite_path))
                        break
            else:
                # Re-run comprehensive to get suite
                suite_li, suite_all, suite_relevant, fname = run_comprehensive_evaluation(
                    config, poll_data_name_vec[0], use_data_type_field, use_only_relevant_rules
                )
                save_results(config, suite_li, fname, suite_all, suite_relevant)
            
            # Run big test
            stats = run_big_test(config, poll_data_big_name, suite_all, fname)
            
            print(f"Execution time: {stats['total_execution_time']:.2f} seconds")
            print(f"Rules run: {stats['run_rule']}, Rules not run: {stats['not_run_rule']}")
            print(f"Total rules: {stats['rule_all']}")
            
            run_big_li.append(stats)
    
    # Print summary
    if run_big_li:
        print("\n" + "=" * 60)
        print("Summary: Big Dataset Results")
        print("=" * 60)
        for stats in run_big_li:
            print(f"\n{stats['pollDataBigName']}:")
            print(f"  Rows: {stats['total_rows']}")
            print(f"  Execution time: {stats['total_execution_time']:.2f} seconds")
            print(f"  Rules run: {stats['run_rule']}")


if __name__ == '__main__':
    main()