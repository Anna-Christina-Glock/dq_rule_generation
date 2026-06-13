"""
Parallel Evaluation Module for Great Expectations

This module provides parallel execution capabilities for validating large datasets
against Great Expectations rule suites using ThreadPoolExecutor.
"""

import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import great_expectations as gx
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


def create_suite_parts(
    suite_all: gx.ExpectationSuite,
    suite_window_size: int
) -> list:
    """
    Split a large expectation suite into smaller parts for parallel processing.
    
    Args:
        suite_all: The complete expectation suite to split
        suite_window_size: Maximum number of expectations per partition
        
    Returns:
        List of expectation suite partitions
    """
    expectations = suite_all.expectations
    total_expectations = len(expectations)
    suite_list = []
    
    for start_idx in range(0, total_expectations, suite_window_size):
        end_idx = min(start_idx + suite_window_size, total_expectations)
        
        # Create suite partition
        suite_part = gx.ExpectationSuite(
            name=f"suite_part_{start_idx}"
        )
        
        # Add expectations to partition
        for expectation in expectations[start_idx:end_idx]:
            suite_part.add_expectation(expectation)
        
        suite_list.append(suite_part)
        
        # Periodic cleanup to prevent memory buildup
        if len(suite_list) % 10 == 0:
            gc.collect()
    
    return suite_list


def validate_single_suite(args: tuple) -> dict:
    """
    Validate a single suite partition against a batch.
    
    Args:
        args: Tuple of (suite_part, batch, res_format, window_id, suite_id)
        
    Returns:
        Dictionary with validation results and metadata
    """
    suite_part, batch, res_format, window_id, suite_id = args
    
    start_time = time.time()
    
    try:
        validation_result = batch.validate(suite_part, result_format=res_format)
        end_time = time.time()
        
        return {
            'window_id': window_id,
            'suite_id': suite_id,
            'result': validation_result,
            'execution_time': end_time - start_time,
            'error': None
        }
    except Exception as e:
        end_time = time.time()
        
        return {
            'window_id': window_id,
            'suite_id': suite_id,
            'error': str(e),
            'result': None,
            'execution_time': end_time - start_time
        }


def validate_batch_parallel(
    df: pd.DataFrame,
    suite: gx.ExpectationSuite,
    suite_window_size: int = 10,
    max_workers: int = 10,
    result_format: dict = None
) -> dict:
    """
    Validate a dataframe against a suite using parallel execution.
    
    Args:
        df: DataFrame to validate
        suite: ExpectationSuite to validate against
        suite_window_size: Number of expectations per partition
        max_workers: Maximum number of parallel workers
        result_format: Format for validation results
        
    Returns:
        Dictionary with validation results, timing info, and statistics
    """
    if result_format is None:
        result_format = {
            "result_format": "COMPLETE",
            "include_unexpected_rows": True
        }
    
    # Get batch
    from evaluate_rules import get_batch
    context, batch = get_batch(df)
    
    # Split suite into partitions
    suite_list = create_suite_parts(suite, suite_window_size)
    
    # Prepare validation tasks
    validation_tasks = [
        (suite_part, batch, result_format, 0, idx)
        for idx, suite_part in enumerate(suite_list)
    ]
    
    # Execute validations in parallel
    results = []
    execution_times = []
    error_count = 0
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(validate_single_suite, task): task
            for task in validation_tasks
        }
        
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                
                if result['error'] is None:
                    success_count += 1
                    results.append(result)
                    execution_times.append(result['execution_time'])
                else:
                    error_count += 1
                    logger.error(
                        f"Validation failed for window {result['window_id']}, "
                        f"suite {result['suite_id']}: {result['error']}"
                    )
                    
            except Exception as exc:
                error_count += 1
                logger.error(f"Validation task raised exception: {exc}")
    
    # Aggregate results
    total_time = sum(execution_times) if execution_times else 0
    avg_time = total_time / len(execution_times) if execution_times else 0
    
    return {
        'results': results,
        'success_count': success_count,
        'error_count': error_count,
        'total_execution_time': total_time,
        'average_execution_time': avg_time,
        'num_suites': len(suite_list)
    }


def validate_large_dataset(
    df: pd.DataFrame,
    suite: gx.ExpectationSuite,
    window_size: int = 1000,
    suite_window_size: int = 10,
    max_workers: int = 10,
    result_format: dict = None
) -> dict:
    """
    Validate a large dataset in windows with parallel suite execution.
    
    Args:
        df: Large DataFrame to validate
        suite: ExpectationSuite to validate against
        window_size: Number of rows per window
        suite_window_size: Number of expectations per partition
        max_workers: Maximum number of parallel workers
        result_format: Format for validation results
        
    Returns:
        Dictionary with per-window validation statistics
    """
    if result_format is None:
        result_format = {
            "result_format": "COMPLETE",
            "include_unexpected_rows": True
        }
    
    total_rows = len(df)
    window_count = (total_rows + window_size - 1) // window_size
    
    all_window_stats = []
    all_suite_stats = {
        'success_count': 0,
        'error_count': 0,
        'total_time': 0,
        'rule_count': 0,
        'not_run_count': 0
    }
    
    for i in range(0, total_rows, window_size):
        end = min(i + window_size, total_rows)
        window = df.iloc[i:end]
        
        logger.info(f"Processing window {i} to {end} ({len(window)} rows)")
        
        window_start = time.time()
        
        # Validate with parallel execution
        window_result = validate_batch_parallel(
            window, suite, suite_window_size, max_workers, result_format
        )
        
        window_end = time.time()
        
        # Extract rule statistics from results
        rule_count = 0
        not_run_count = 0
        
        for suite_result in window_result['results']:
            if suite_result['result'] is not None:
                for res in suite_result['result'].get('results', []):
                    result_data = res.get('result', {})
                    if result_data:
                        rule_count += 1
                    else:
                        not_run_count += 1
        
        window_stats = {
            'window_start': i,
            'window_end': end,
            'row_count': len(window),
            'execution_time': window_end - window_start,
            'suite_success': window_result['success_count'],
            'suite_errors': window_result['error_count'],
            'rule_count': rule_count,
            'not_run_count': not_run_count,
            'avg_suite_time': window_result['average_execution_time']
        }
        all_window_stats.append(window_stats)
        
        # Aggregate suite statistics
        all_suite_stats['success_count'] += window_result['success_count']
        all_suite_stats['error_count'] += window_result['error_count']
        all_suite_stats['total_time'] += window_result['total_execution_time']
        all_suite_stats['rule_count'] += rule_count
        all_suite_stats['not_run_count'] += not_run_count
    
    return {
        'window_stats': all_window_stats,
        'suite_stats': all_suite_stats,
        'total_rows': total_rows,
        'window_count': window_count
    }