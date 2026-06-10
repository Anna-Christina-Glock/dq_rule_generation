"""
Rule Evaluation Module for Great Expectations

This module provides functions for evaluating LLM-generated Great Expectations rules.
It includes validation, precision/recall calculation, and filtering pipeline functions.
"""

import traceback
import great_expectations as gx
from pathlib import Path
import pandas as pd
import re
import json
import itertools
import logging
from functools import reduce
import operator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_batch(
    df: pd.DataFrame,
    data_source_name: str = "data_source",
    asset_name: str = "data_asset",
    batch_definition_name: str = "batch"
) -> tuple:
    """
    Create a Great Expectations context and batch for validation.
    
    Args:
        df: DataFrame to create batch from
        data_source_name: Name for the data source
        asset_name: Name for the data asset
        batch_definition_name: Name for the batch definition
        
    Returns:
        Tuple of (context, batch)
    """
    context = gx.get_context()
    
    # Clean up existing data source if it exists
    if data_source_name in [ds["name"] for ds in context.list_datasources()]:
        logger.info(f"Deleting existing datasource: '{data_source_name}'")
        context.delete_datasource(name=data_source_name)
    
    # Add new data source and asset
    data_source = context.data_sources.add_pandas(name=data_source_name)
    data_asset = data_source.add_dataframe_asset(name=asset_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
    
    batch_parameters = {"dataframe": df}
    batch = (
        context.data_sources.get(data_source_name)
        .get_asset(asset_name)
        .get_batch_definition(batch_definition_name)
        .get_batch(batch_parameters=batch_parameters)
    )
    
    return context, batch


def calculate_precision_recall(true_err_idx: list, pred_err_idx: list) -> dict:
    """
    Calculate precision and recall based on true and predicted error indices.
    
    Args:
        true_err_idx: List of true error indices
        pred_err_idx: List of predicted error indices
        
    Returns:
        Dictionary with precision, recall, TP, FP, FN
    """
    true_set = set(true_err_idx)
    pred_set = set(pred_err_idx)
    
    TP = len(true_set & pred_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'TP': TP,
        'FP': FP,
        'FN': FN
    }


def get_validation_result(
    validation_results: dict,
    df_res_all: pd.DataFrame,
    is_one_row: bool = False,
    err_col: str = None
) -> list:
    """
    Extract validation results from Great Expectations output.
    
    Args:
        validation_results: Results from batch.validate()
        df_res_all: Original dataframe with error information
        is_one_row: Whether this is for a single row
        err_col: Error column name for filtering
        
    Returns:
        List of validation result dictionaries
    """
    res_val_li = []
    
    if is_one_row:
        err_col_li = [item.split('<->') for item in df_res_all.loc['errCol'].split(';')[1:]]
        err_col_li = list(itertools.chain.from_iterable(err_col_li))
    else:
        if ';' in err_col:
            err_col_li = [err_col.split(';')[1]]
        else:
            err_col_li = [err_col] if err_col else []
    
    if is_one_row:
        err_rule = df_res_all.loc['errRule'].split(';')[1:]
    else:
        err_rule = ''
    
    for val_res in validation_results.get('results', []):
        # Get column names from expectation
        col_key = [k for k in val_res['expectation_config']['kwargs'].keys() if 'column' in k]
        col_name = [val_res['expectation_config']['kwargs'][x] for x in col_key]
        
        rule_type = val_res['expectation_config']['type']
        description = val_res['expectation_config'].get('description')
        
        # Extract result information
        num_wrong = 0
        wrong_vals = None
        wrong_idx = None
        
        if val_res['result']:
            num_wrong = val_res['result'].get('unexpected_count', 0) or 0
            num_wrong += val_res['result'].get('missing_count', 0) or 0
            wrong_vals = val_res['result'].get('unexpected_list')
            wrong_idx = val_res['result'].get('unexpected_index_list')
        
        # Check if this expectation matches the error columns
        col_in_list = any([cn in err_col_li for cn in col_name])
        
        if isinstance(col_name, list):
            col_name = "', '".join(str(c) for c in col_name)
        
        if col_in_list:
            err_col_idx_true = [i for i, x in enumerate(err_col_li) if x in col_name]
            num_wrong_true = len(df_res_all.query('isDirty')) if not is_one_row else 1
            
            if not val_res['result']:
                err_col_idx = []
            else:
                try:
                    err_col_idx = val_res['result'].get('unexpected_index_list', [])
                except KeyError:
                    err_col_idx = []
        else:
            err_col_idx = []
            err_col_idx_true = []
            num_wrong_true = 0
        
        res_val_li.append({
            'colname': col_name,
            'rule_type': rule_type,
            'description': description,
            'num_wrong': num_wrong,
            'wrong_vals': wrong_vals,
            'wrong_idx': wrong_idx,
            'num_wrong_true': num_wrong_true,
            'err_rule': err_rule,
            'err_col': err_col_li,
            'col_in_list': col_in_list,
            'err_col_idx': err_col_idx,
            'err_col_idx_true': err_col_idx_true
        })
    
    return res_val_li


def get_precision_recall_per_row(val_res_df: pd.DataFrame, all_col_arr: list) -> dict:
    """
    Calculate precision and recall at the row level.
    
    Args:
        val_res_df: Validation results dataframe
        all_col_arr: All column names
        
    Returns:
        Dictionary with precision, recall, TP, FP, FN at row level
    """
    true_err_col = val_res_df['errCol'].iloc[0] if 'errCol' in val_res_df.columns else []
    true_non_err_col = [col for col in all_col_arr if col not in true_err_col]
    
    tmp_df = val_res_df[val_res_df['wrongVals'].notna()]
    pred_err_col = [x.replace("'", "").split(', ') for x in tmp_df[tmp_df['numWrong'] > 0]['colname'].values]
    pred_err_col = list(itertools.chain(*pred_err_col))
    pred_non_err_col = [col for col in all_col_arr if col not in pred_err_col]
    
    precision, recall, TP, FP, FN = calculate_precision_recall(true_err_col, pred_err_col)
    
    return {
        'precision_per_row': precision,
        'recall_per_row': recall,
        'TP_per_row': TP,
        'FP_per_row': FP,
        'FN_per_row': FN
    }


def get_precision_recall_per_rule_col(val_res: list) -> dict:
    """
    Calculate precision and recall at the rule/column level.
    
    Args:
        val_res: Validation results list
        
    Returns:
        Dictionary with precision, recall, TP, FP, FN at rule/column level
    """
    err_col_idx_all = []
    wrong_idx_all = []
    
    for row in val_res:
        for i, col_name in enumerate(row.get('colname', [])):
            suffix = f'_{str(col_name).replace("'", "")}'
            
            err_idx = [str(e) + suffix for e in row.get('errColIdx_true', [[]])[i]]
            err_col_idx_all.extend(err_idx)
            
            wrong_idx = row.get('wrongIdxLi', [None])[i]
            if wrong_idx:
                wrong_idx = [str(e) + suffix for e in wrong_idx]
            wrong_idx_all.extend(wrong_idx if wrong_idx else [])
    
    precision, recall, TP, FP, FN = calculate_precision_recall(err_col_idx_all, wrong_idx_all)
    
    return {
        'precision_per_rule_col': precision,
        'recall_per_rule_col': recall,
        'TP_per_rule_col': TP,
        'FP_per_rule_col': FP,
        'FN_per_rule_col': FN
    }


def get_precision_recall_per_col(val_res: list) -> dict:
    """
    Calculate precision and recall at the column level.
    
    Args:
        val_res: Validation results list
        
    Returns:
        Dictionary with precision, recall, TP, FP, FN at column level
    """
    err_col_idx_all = []
    wrong_idx_all = []
    
    for row in val_res:
        for i, col_name in enumerate(row.get('colname', [])):
            suffix = f'_{str(col_name).replace("'", "")}'
            
            err_idx = [str(e) + suffix for e in row.get('errColIdx_true', [[]])[i]]
            err_col_idx_all.extend(err_idx)
            
            wrong_idx = row.get('wrongIdxLi', [None])[i]
            if wrong_idx and len(wrong_idx) != row.get('numWrong', [0])[i]:
                wrong_idx = [str(e) + suffix for e in range(0, row.get('numWrong', [0])[i])]
            if wrong_idx:
                wrong_idx = [str(e) + suffix for e in wrong_idx]
            wrong_idx_all.extend(wrong_idx if wrong_idx else [])
    
    precision, recall, TP, FP, FN = calculate_precision_recall(err_col_idx_all, wrong_idx_all)
    
    return {
        'precision_per_col': precision,
        'recall_per_col': recall,
        'TP_per_col': TP,
        'FP_per_col': FP,
        'FN_per_col': FN
    }


def fill_expectation_suite(
    json_str_list: list,
    suite: gx.ExpectationSuite,
    suite_all: gx.ExpectationSuite = None,
    note: str = None
) -> tuple:
    """
    Fill a Great Expectations suite from parsed JSON.
    
    Args:
        json_str_list: List of JSON objects with expectation definitions
        suite: ExpectationSuite to add to
        suite_all: Optional second suite to add to
        note: Optional note to add to expectations
        
    Returns:
        Tuple of (eval_list, suite, exception_count, last_index, suite_all)
    """
    j = -1
    eval_li = []
    exception_count = 0
    
    for gen_code in json_str_list:
        j += 1
        try:
            if isinstance(gen_code, dict):
                gen_code_dict = gen_code
                if len(gen_code.keys()) == 1 and 'code' not in gen_code:
                    gen_code_dict = gen_code[list(gen_code.keys())[0]]
                
                if 'column' in gen_code or 'code' in gen_code:
                    code_str = gen_code.get('code', str(gen_code))
                else:
                    code_str = gen_code_dict.get('code', str(gen_code_dict))
                
                # Execute the expectation code
                eval(f"suite.add_expectation({code_str})")
                
                if note is not None:
                    suite.expectations[-1].notes = note
                
                if suite_all is not None:
                    eval(f"suite_all.add_expectation({code_str})")
                    if note is not None:
                        suite_all.expectations[-1].notes = note
                
                eval_li.append({'j': j, 'can_run': True, 'exception': None})
                
            elif isinstance(gen_code, str):
                # Handle string case
                continue
                
        except Exception as e:
            exception_count += 1
            eval_li.append({
                'j': j,
                'can_run': False,
                'exception': str(e)
            })
            logger.error(f"Error adding expectation {j}: {e}")
    
    return eval_li, suite, exception_count, j, suite_all


def extract_expectations_from_json(
    answer_str: str,
    json_marker: str = "great_expectations_final"
) -> list:
    """
    Extract expectations from LLM response JSON.
    
    Args:
        answer_str: LLM response string
        json_marker: Marker string to find in response
        
    Returns:
        List of expectation dictionaries
    """
    # Try to extract JSON from response
    json_patterns = [
        r'```json\s*\{(\s*)"great_expectations_final"',
        r'```json\s*\{(\s*)"great_expectations_final"',
        r'"great_expectations_final"'
    ]
    
    extract_fin_json = answer_str
    
    for pattern in json_patterns:
        match = re.search(pattern, answer_str)
        if match:
            # Split at the marker
            split_arr = re.split(pattern, answer_str)
            if len(split_arr) > 1:
                extract_fin_json = split_arr[-1]
            break
    
    if not extract_fin_json or len(extract_fin_json) <= 1:
        return []
    
    # Clean up JSON string
    json_str = extract_fin_json.strip()
    json_str = json_str.replace('# Final Json Format', '')
    json_str = re.sub(r'\n\]}\n}', '\n}]\n}', json_str)
    json_str = re.sub(r'\n\]}\n', '\n}]\n', json_str)
    
    try:
        json_data = json.loads(json_str)
        expectations = json_data.get('great_expectations_final', [])
        
        if isinstance(expectations, dict):
            # Handle single expectation case
            key_val = list(expectations.keys())[0]
            if isinstance(expectations[key_val], dict):
                expectations = [expectations[key_val]]
            else:
                expectations = expectations[key_val]
        elif isinstance(expectations, list) and len(expectations) > 0:
            # Handle list case - extract inner dict if needed
            if isinstance(expectations[0], dict):
                if 'code' in expectations[0]:
                    pass  # Already in correct format
                elif list(expectations[0].keys())[0] in ['column', 'code']:
                    pass  # Already in correct format
                else:
                    key = list(expectations[0].keys())[0]
                    expectations = expectations[0][key] if isinstance(expectations[0][key], list) else [expectations[0][key]]
        
        return expectations if isinstance(expectations, list) else [expectations]
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return []


def validate_rule_on_dataframe(
    df: pd.DataFrame,
    expectations: list,
    suite_name: str = "validation_suite",
    result_format: dict = None
) -> dict:
    """
    Validate a dataframe against a list of expectations.
    
    Args:
        df: DataFrame to validate
        expectations: List of expectation definitions
        suite_name: Name for the expectation suite
        result_format: Format for validation results
        
    Returns:
        Validation results dictionary
    """
    if result_format is None:
        result_format = {
            "result_format": "COMPLETE",
            "include_unexpected_rows": True
        }
    
    # Create suite
    suite = gx.ExpectationSuite(name=suite_name)
    
    # Fill suite with expectations
    _, suite, exception_count, j, _ = fill_expectation_suite(expectations, suite)
    
    if exception_count > 0:
        return {
            'can_run': False,
            'exception_count': exception_count,
            'validation_results': None
        }
    
    # Get batch
    try:
        context, batch = get_batch(df)
        
        # Validate
        validation_results = batch.validate(suite, result_format=result_format)
        
        return {
            'can_run': True,
            'suite': suite,
            'validation_results': validation_results,
            'exception_count': exception_count,
            'num_expectations': j + 1
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            'can_run': False,
            'exception_count': exception_count,
            'exception': str(e),
            'validation_results': None
        }