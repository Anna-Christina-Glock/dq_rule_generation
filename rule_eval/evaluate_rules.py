"""
Rule Evaluation Module for Great Expectations

This module provides functions for evaluating LLM-generated Great Expectations rules.
It includes validation, precision/recall calculation, filtering pipeline functions,
data type correction, suite persistence, and comprehensive evaluation.
"""

import datetime
import gc
import itertools
import json
import logging
import operator
import re
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import great_expectations as gx
import pandas as pd

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
    """Create a Great Expectations dataframe batch.

    The datasource is replaced on each call because GX batches retain their
    dataframe by reference. Reusing an existing datasource can otherwise
    validate a later suite against stale data.
    """
    context = gx.get_context()
    existing = {source["name"] for source in context.list_datasources()}
    if data_source_name in existing:
        context.delete_datasource(name=data_source_name)

    data_source = context.data_sources.add_pandas(name=data_source_name)
    asset = data_source.add_dataframe_asset(name=asset_name)
    asset.add_batch_definition_whole_dataframe(batch_definition_name)
    batch = (
        context.data_sources.get(data_source_name)
        .get_asset(asset_name)
        .get_batch_definition(batch_definition_name)
        .get_batch(batch_parameters={"dataframe": df})
    )
    return context, batch


def validate_with_typed_between(
    df: pd.DataFrame,
    suite: gx.ExpectationSuite,
    result_format: dict,
) -> dict:
    """Validate a suite while preserving numeric/date semantics for between rules."""
    between = [
        expectation
        for expectation in suite.expectations
        if expectation.expectation_type == "expect_column_values_to_be_between"
    ]
    if not between:
        _, batch = get_batch(df)
        return batch.validate(suite, result_format=result_format)

    typed_df = df.copy()
    between_suite = gx.ExpectationSuite(name=f"{suite.name}_between")
    other_suite = gx.ExpectationSuite(name=f"{suite.name}_other")
    for expectation in suite.expectations:
        expectation_copy = expectation.copy()
        if expectation_copy.expectation_type != "expect_column_values_to_be_between":
            other_suite.add_expectation(expectation_copy)
            continue

        bounds = [
            bound
            for bound in (expectation_copy.min_value, expectation_copy.max_value)
            if bound is not None
        ]
        is_date = any(
            isinstance(bound, (datetime.date, datetime.datetime, pd.Timestamp))
            for bound in bounds
        )
        if is_date:
            try:
                typed_df[expectation_copy.column] = pd.to_datetime(
                    typed_df[expectation_copy.column], errors="coerce"
                )
            except ValueError:
                typed_df[expectation_copy.column] = pd.to_datetime(
                    typed_df[expectation_copy.column], errors="coerce", utc=True
                ).dt.tz_localize(None)
            if expectation_copy.min_value is not None:
                expectation_copy.min_value = pd.Timestamp(expectation_copy.min_value)
            if expectation_copy.max_value is not None:
                expectation_copy.max_value = pd.Timestamp(expectation_copy.max_value)
        else:
            typed_df[expectation_copy.column] = pd.to_numeric(
                typed_df[expectation_copy.column], errors="coerce"
            )
        between_suite.add_expectation(expectation_copy)

    results = []
    statistics = {
        "evaluated_expectations": 0,
        "successful_expectations": 0,
        "unsuccessful_expectations": 0,
    }
    for frame, partition, source_name in (
        (df, other_suite, "validation_other"),
        (typed_df, between_suite, "validation_between"),
    ):
        if not partition.expectations:
            continue
        _, batch = get_batch(frame, data_source_name=source_name)
        validation = batch.validate(partition, result_format=result_format)
        results.extend(validation["results"])
        for key in statistics:
            statistics[key] += validation["statistics"].get(key, 0)

    statistics["success"] = all(result.get("success") for result in results)
    return {
        "success": statistics["success"],
        "results": results,
        "statistics": statistics,
    }


def calculate_precision_recall(
    true_err_idx: list,
    pred_err_idx: list,
) -> tuple[float, float, int, int, int]:
    """Return precision, recall, TP, FP, and FN for two cell-index sets."""
    true_set = set(true_err_idx)
    pred_set = set(pred_err_idx)
    true_positive = len(true_set & pred_set)
    false_positive = len(pred_set - true_set)
    false_negative = len(true_set - pred_set)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    return precision, recall, true_positive, false_positive, false_negative


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
        # Match original behavior: list length must match dataframe length for index alignment
        df_len = len(df_res_all)
        if ';' in err_col:
            col_val = err_col.split(';')[1]
        else:
            col_val = err_col
        err_col_li = [col_val] * df_len if col_val else []
    
    if is_one_row:
        err_rule = df_res_all.loc['errRule'].split(';')[1:]
    else:
        err_rule = ''
    
    for val_res in validation_results.get('results', []):
        col_key = [k for k in val_res['expectation_config']['kwargs'].keys() if 'column' in k]
        col_name = [val_res['expectation_config']['kwargs'][x] for x in col_key]
        
        rule_type = val_res['expectation_config']['type']
        description = val_res['expectation_config'].get('description')
        
        num_wrong = 0
        wrong_vals = None
        wrong_idx = None
        
        if val_res['result']:
            num_wrong = val_res['result'].get('unexpected_count', 0) or 0
            num_wrong += val_res['result'].get('missing_count', 0) or 0
            wrong_vals = val_res['result'].get('unexpected_list')
            wrong_idx = val_res['result'].get('unexpected_index_list')
        
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
    true_err_col = val_res_df["err_col"].iloc[0] if "err_col" in val_res_df else []
    true_non_err_col = [col for col in all_col_arr if col not in true_err_col]
    
    tmp_df = val_res_df[val_res_df['wrong_vals'].notna()]
    pred_err_col = [x.replace("'", "").split(', ') for x in tmp_df[tmp_df['num_wrong'] > 0]['colname'].values]
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
            suffix = '_' + str(col_name).replace("'", "")
            
            err_idx = [str(e) + suffix for e in row.get('err_col_idx_true', [[]])[i]]
            err_col_idx_all.extend(err_idx)
            
            wrong_idx = row.get('wrong_idx', [None])[i]
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
            suffix = '_' + str(col_name).replace("'", "")
            
            err_idx = [str(e) + suffix for e in row.get('err_col_idx_true', [[]])[i]]
            err_col_idx_all.extend(err_idx)
            
            wrong_idx = row.get('wrong_idx', [None])[i]
            if wrong_idx and len(wrong_idx) != row.get('num_wrong', [0])[i]:
                wrong_idx = [str(e) + suffix for e in range(0, row.get('num_wrong', [0])[i])]
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
                
                eval(f"suite.add_expectation({code_str})")
                
                if note is not None:
                    suite.expectations[-1].notes = note
                
                if suite_all is not None:
                    eval(f"suite_all.add_expectation({code_str})")
                    if note is not None:
                        suite_all.expectations[-1].notes = note
                
                eval_li.append({'j': j, 'can_run': True, 'exception': None})
                
            elif isinstance(gen_code, str):
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


def _flatten_expectations(value: Any) -> list[dict]:
    """Return every nested expectation object that contains executable code."""
    if isinstance(value, dict):
        if isinstance(value.get("code"), str):
            return [value]
        flattened = []
        for child in value.values():
            flattened.extend(_flatten_expectations(child))
        return flattened
    if isinstance(value, list):
        flattened = []
        for child in value:
            flattened.extend(_flatten_expectations(child))
        return flattened
    return []


def extract_expectations_from_json(
    answer_str: str,
    json_marker: str = "great_expectations_final",
) -> list[dict]:
    """Extract executable expectation objects from an LLM response."""
    if not isinstance(answer_str, str) or json_marker not in answer_str:
        return []

    decoder = json.JSONDecoder()
    variants = [
        answer_str.replace("# Final Json Format", ""),
        answer_str.replace("# Final Json Format", "")
        .replace("\n]}\n}", "\n}]\n}")
        .replace("\n]}\n", "\n}]\n"),
    ]
    for text in variants:
        for start in (index for index, char in enumerate(text) if char == "{"):
            try:
                payload, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and json_marker in payload:
                return _flatten_expectations(payload[json_marker])

    logger.warning("Could not parse %s from model response", json_marker)
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
    
    gx.get_context()
    suite = gx.ExpectationSuite(name=suite_name)
    _, suite, exception_count, j, _ = fill_expectation_suite(expectations, suite)
    
    if exception_count > 0:
        return {
            'can_run': False,
            'exception_count': exception_count,
            'validation_results': None
        }
    
    try:
        validation_results = validate_with_typed_between(
            df, suite, result_format=result_format
        )
        
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


# ============================================================================
# Data Type Correction Utilities
# ============================================================================

def to_type(value: Any, dtype: str) -> Any:
    """
    Convert a value to the specified data type.
    
    Args:
        value: Value to convert
        dtype: Target data type (int, float, str, etc.)
        
    Returns:
        Converted value or None if conversion fails
    """
    is_mv = False
    if value == 'nan':
        is_mv = True

    if dtype in ['int', 'integer'] or dtype == 'numeric':
        if is_mv:
            return None
        else:
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return None
    elif dtype == 'float':
        if is_mv:
            return None
        else:
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
    elif dtype in ['str', 'string']:
        if is_mv:
            return None
        else:
            try:
                return str(value)
            except (ValueError, TypeError):
                return None
    else:
        logger.warning(f"Unknown dtype: {dtype}")
        return None


def to_type_df_col(df: pd.DataFrame, colname: str, dtype: str) -> pd.Series:
    """
    Convert a DataFrame column to the specified data type.
    
    Args:
        df: DataFrame containing the column
        colname: Name of the column to convert
        dtype: Target data type
        
    Returns:
        Series with converted values
    """
    return df[colname].apply(lambda x: to_type(x, dtype))


def get_dtype_corr_df(df_dirty: pd.DataFrame, json_str_list) -> pd.DataFrame:
    """
    Apply data type corrections to a DataFrame based on JSON specifications.
    
    Args:
        df_dirty: Dirty DataFrame to correct
        json_str_list: List or dict containing datatype specifications
        
    Returns:
        DataFrame with corrected data types
    """
    if isinstance(json_str_list, list):
        for json_str in json_str_list:
            if isinstance(json_str, dict):
                if 'great_expectations_final' in json_str:
                    json_str = json_str['great_expectations_final']
                    if isinstance(json_str, dict):
                        for key in json_str.keys():
                            json_str1 = json_str[key]
                            dtype = json_str1.get('datatype', '')
                            col_name = json_str1.get('column', '')
                            _apply_dtype_correction(df_dirty, col_name, dtype)
                    else:
                        for item in json_str:
                            if isinstance(item, dict):
                                dtype = item.get('datatype', '')
                                col_name = item.get('column', '')
                                _apply_dtype_correction(df_dirty, col_name, dtype)
                else:
                    for key in json_str.keys():
                        if isinstance(json_str[key], dict):
                            dtype = json_str[key].get('datatype', '')
                            col_name = json_str[key].get('column', '')
                            _apply_dtype_correction(df_dirty, col_name, dtype)
            else:
                if isinstance(json_str, dict):
                    dtype = json_str.get('datatype', '')
                    col_name = json_str.get('column', '')
                    _apply_dtype_correction(df_dirty, col_name, dtype)
                    
    elif isinstance(json_str_list, dict):
        if 'great_expectations_final' in json_str_list:
            json_str = json_str_list['great_expectations_final']
            if isinstance(json_str, dict):
                for key in json_str.keys():
                    json_str1 = json_str[key]
                    dtype = json_str1.get('datatype', '')
                    col_name = json_str1.get('column', '')
                    _apply_dtype_correction(df_dirty, col_name, dtype)
            else:
                for item in json_str:
                    if isinstance(item, dict):
                        dtype = item.get('datatype', '')
                        col_name = item.get('column', '')
                        _apply_dtype_correction(df_dirty, col_name, dtype)
        else:
            for key in json_str_list.keys():
                json_str = json_str_list[key]
                if isinstance(json_str, dict):
                    dtype = json_str.get('datatype', '')
                    col_name = json_str.get('column', '')
                    _apply_dtype_correction(df_dirty, col_name, dtype)
    
    return df_dirty


def _apply_dtype_correction(df: pd.DataFrame, col_name: str, dtype: str) -> None:
    """
    Apply data type correction to a single column (helper function).
    
    Args:
        df: DataFrame to modify
        col_name: Column name to correct
        dtype: Target data type
    """
    if col_name == 'stra\\u00dfe':
        col_name = 'straße'
    
    if col_name not in df.columns:
        return
    
    if dtype:
        try:
            df.loc[:, col_name] = to_type_df_col(df, col_name, dtype)
        except Exception as e:
            logger.warning(f"Could not correct dtype for {col_name}: {e}")


# ============================================================================
# Suite Persistence Functions
# ============================================================================

def save_suite_as_json(suite: gx.ExpectationSuite, path_str: str) -> None:
    """
    Save an expectation suite as a JSON file.
    
    Args:
        suite: ExpectationSuite to save
        path_str: Path to save the JSON file
    """
    export_path = Path(path_str)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    suite_dict = suite.to_json_dict()
    export_path.write_text(json.dumps(suite_dict, indent=2))


def get_suite_from_json(suite_json_path: str) -> gx.ExpectationSuite:
    """
    Load an expectation suite from a JSON file.
    
    Args:
        suite_json_path: Path to the JSON file
        
    Returns:
        Loaded ExpectationSuite
    """
    suite_dict = json.loads(Path(suite_json_path).read_text(encoding="utf-8"))
    return gx.ExpectationSuite(**suite_dict)


# ============================================================================
# Comprehensive Evaluation Function (check_all_rules equivalent)
# ============================================================================

def check_all_rules(
    df_res_all: pd.DataFrame,
    all_col_count: int = 14,
    use_only_relevant_rules: bool = False,
    suite_relevant: gx.ExpectationSuite = None,
    row_range: Optional[Tuple[int, int]] = None,
) -> tuple:
    """
    Validate all rules from a dataframe of error specifications.
    
    This is the main evaluation function that processes each row in the input
    dataframe, extracts expectations from LLM responses, and validates them
    against the corresponding data.
    
    Args:
        df_res_all: DataFrame with columns ['errCol', 'errRule', 'answer', 'isDirty']
        all_col_count: Number of data columns in the dataframe
        use_only_relevant_rules: Whether to use only relevant rules for evaluation
        suite_relevant: Optional existing suite for relevant rules
        
    Returns:
        Tuple of (suite_list, suite_all, suite_relevant)
    """
    get_batch(df_res_all.iloc[:, 0:all_col_count])
    res_format = {
        "result_format": "COMPLETE",
        "include_unexpected_rows": True,
    }

    suite_li = []
    suite_all = gx.ExpectationSuite(name="suite_with_all_rules")
    
    if suite_relevant is None:
        suite_relevant = gx.ExpectationSuite(name="suite_with_relevant_rules")
    
    if row_range:
        start_row, end_row = row_range
        indices = range(start_row, min(end_row, len(df_res_all)))
    else:
        indices = range(len(df_res_all))

    for i in indices:
        df_row = df_res_all.iloc[[i]]
        df_dirty = df_row.iloc[:, 0:all_col_count]
        
        expectation_suite_name = f"vllm_test_manuell_pk_{df_row.iloc[0].loc['pk']}"
        suite = gx.ExpectationSuite(name=expectation_suite_name)
        ans_str = df_row.iloc[0].loc['answer']
        
        json_str_list = extract_expectations_from_json(ans_str)
        if not json_str_list:
            suite_dict = {
                "i": i,
                "suite": None,
                "val_res": None,
                "validation_results_clean": None,
            }
            suite_dict.update(df_row.iloc[0].to_dict())
            suite_dict.update(
                {"rule_count": 0, "exception_count": 0, "exception_li": []}
            )
            suite_dict.update(_get_default_precision_recall())
            suite_li.append(suite_dict)
            continue
        
        j = -1
        eval_li = []
        exception_count = 0
        
        note_str = f"{df_res_all.iloc[i, :].loc[['errCol']].values[0]}-{df_res_all.iloc[i, :].loc[['errRule']].values[0]}"
        eval_li, suite, exception_count, j, suite_all = fill_expectation_suite(
            json_str_list, suite, suite_all, note=note_str
        )
        
        try:
            validation_results_clean = validate_with_typed_between(
                df_dirty, suite, result_format=res_format
            )
        except Exception as e:
            suite_dict = {
                'i': i, 'suite': suite, 'val_res': None,
                'validation_results_clean': None, 'can_run_suite': False, 'Exception': e
            }
            suite_dict.update(df_row.iloc[0].to_dict())
            suite_dict.update({
                'rule_count': j + 1, 'exception_count': exception_count, 'exception_li': eval_li
            })
            suite_dict.update(_get_default_precision_recall())
            suite_li.append(suite_dict)
            continue
        
        val_res_df = pd.DataFrame(get_validation_result(
            validation_results_clean, df_res_all.iloc[i, :], is_one_row=True
        ))
        
        suite_dict = {
            'i': i, 'suite': suite, 'val_res': val_res_df,
            'validation_results_clean': validation_results_clean, 'can_run_suite': True,
            'Exception': None
        }
        suite_dict.update(df_row.iloc[0].to_dict())
        suite_dict.update({
            'rule_count': j + 1, 'exception_count': exception_count, 'exception_li': eval_li
        })
        
        all_col_arr = df_res_all.iloc[0, 0:all_col_count].index
        
        if len(val_res_df) == 0:
            suite_dict.update(_get_default_precision_recall())
        else:
            suite_dict.update(get_precision_recall_from_validation(val_res_df, all_col_arr))
        
        err_col = df_row.loc[:, 'errCol'].iloc[0]
        err_rule = df_row.loc[:, 'errRule'].iloc[0]
        
        # Per-rule-column evaluation
        val_li = list()
        val_li_col = list()
        
        for i_sp in range(len(err_col.split(';')) - 1):
            err_col_i_sp = err_col.split(';')[i_sp + 1]
            err_rule_i_sp = err_rule.split(';')[i_sp + 1]
            err_col_str_li = err_col.split(';')[i_sp + 1].split('<->')
            err_rule_str_li = (err_rule.split(';')[i_sp + 1].split('<->')) * len(err_col_str_li)
            
            for j_sp in range(len(err_col_str_li)):
                err_col_val = f'{err_col_str_li[j_sp]}'
                err_rule_val = f'{err_rule_str_li[j_sp]}'
                logger.info(f"i:{i}/{len(df_res_all)}: {err_col_val} - {err_rule_val}")
                
                df_err_col = df_res_all.query(
                    f'errCol.str.contains("{err_col_i_sp}")'
                ).query(f'errRule.str.contains("{err_rule_i_sp}")')
                
                # Compare positions
                df_err_col = _compare_positions(df_err_col, err_col_i_sp, err_rule_i_sp)
                
                df_clean_col = df_res_all.query(
                    f'not errCol.str.contains("{err_col_i_sp}")'
                ).copy()
                df_clean_col.loc[:, "isDirty"] = False
                df_clean_col.loc[:, "errCol"] = ""
                df_clean_col.loc[:, "errRule"] = ""
                df_col_all = pd.concat(
                    [df_err_col, df_clean_col],
                    ignore_index=True,
                    sort=False,
                )
                
                suite_per_rule_col = gx.ExpectationSuite(name='eval_perRule_perCol')
                
                if use_only_relevant_rules:
                    if len(val_res_df) > 0:
                        val_res_df_part = val_res_df.query("num_wrong == num_wrong_true")
                        if len(val_res_df_part) == 0:
                            continue
                        rules_types = val_res_df_part['rule_type']
                        rule_cols = [x.replace("'", "") for x in val_res_df_part['colname'].values]
                    else:
                        rules_types = list()
                    
                    for e in suite.expectations:
                        err_type = e.expectation_type
                        if not(err_type in rules_types.values):
                            continue
                        
                        if 'column' in e.args_keys:
                            if e.column in err_col_val and e.column in rule_cols:
                                suite_per_rule_col.add_expectation(e)
                                suite_relevant.add_expectation(e)
                        elif 'column_list' in e.args_keys:
                            if any([x in err_col_val for x in e.column_list]) and any([x in rule_cols for x in e.column_list]):
                                suite_per_rule_col.add_expectation(e)
                                suite_relevant.add_expectation(e)
                        elif 'column_a' in e.args_keys:
                            if (e.column_a in err_col_val or e.column_b in err_col_val) and (e.column_a in rule_cols or e.column_b in rule_cols):
                                suite_per_rule_col.add_expectation(e)
                                suite_relevant.add_expectation(e)
                        else:
                            suite_per_rule_col.add_expectation(e)
                            suite_relevant.add_expectation(e)
                else:
                    suite_per_rule_col = suite
                
                if len(suite_per_rule_col.expectations) == 0:
                    val_dict = {
                        'err_col': [err_col_val], 'err_rule': err_rule_val,
                        'num_similar_rows': len(df_err_col)
                    }
                    val_dict.update({
                        'wrong_vals': [], 'num_wrong': [0], 'num_wrong_true': [len(df_err_col)],
                        'wrong_idx_li': [], 'col_in_list': [], 'colname': [err_col_val],
                        'err_col_idx_true': [[0]], 'description': [None], 'err_col_idx': []
                    })
                    val_li.append(val_dict)
                    val_li_col.append(val_dict)
                    continue
                
                validation_results_clean_all = validate_with_typed_between(
                    df_err_col.reset_index(),
                    suite_per_rule_col,
                    result_format=res_format,
                )
                validation_results_clean_col = validate_with_typed_between(
                    df_col_all.reset_index(),
                    suite_per_rule_col,
                    result_format=res_format,
                )
                val_res_dict_all = get_validation_result(validation_results_clean_all, df_err_col.reset_index(), is_one_row=False, err_col=err_col_val)
                val_res_dict_all_col = get_validation_result(validation_results_clean_col, df_col_all.reset_index(), is_one_row=False, err_col=err_col_val)
                
                val_dict = {
                    'err_col': err_col_val, 'err_rule': err_rule_val,
                    'num_similar_rows': len(df_err_col),
                    'validation_results_clean_all': validation_results_clean_all,
                    'val_res_dict': val_res_dict_all
                }
                val_dict_col = {
                    'err_col': err_col_val, 'err_rule': err_rule_val,
                    'num_similar_rows': len(df_err_col),
                    'validation_results_clean_all': validation_results_clean_col,
                    'val_res_dict': val_res_dict_all_col
                }
                
                # Combine validation results
                all_keys = reduce(operator.or_, (d.keys() for d in val_res_dict_all))
                vol_res_dict_comb = {key: [d.get(key) for d in val_res_dict_all] for key in all_keys}
                val_dict.update(vol_res_dict_comb)
                val_li.append(val_dict)
                
                all_keys = reduce(operator.or_, (d.keys() for d in val_res_dict_all_col))
                vol_res_dict_comb = {key: [d.get(key) for d in val_res_dict_all_col] for key in all_keys}
                val_dict_col.update(vol_res_dict_comb)
                val_li_col.append(val_dict_col)
        
        # Update suite dict with per-rule-column metrics
        if len(val_li) == 0:
            suite_dict.update({'val_li': None})
            suite_dict.update(_get_default_precision_recall_per_rule_col())
        else:
            suite_dict.update({'val_li': val_li})
            if len(suite_per_rule_col.expectations) == 0:
                suite_dict.update(_get_default_precision_recall_per_rule_col())
            else:
                suite_dict.update(get_precision_recall_from_validation_per_rule_col(val_li))
        
        if len(val_li_col) == 0:
            suite_dict.update({'val_li_col': None})
            suite_dict.update(_get_default_precision_recall_per_col())
        else:
            suite_dict.update({'val_li_col': val_li_col})
            if len(suite_per_rule_col.expectations) == 0:
                suite_dict.update(_get_default_precision_recall_per_col())
            else:
                suite_dict.update(get_precision_recall_from_validation_per_col(val_li_col))
        
        suite_li.append(suite_dict)
        gc.collect()
    
    return suite_li, suite_all, suite_relevant


def _get_default_precision_recall() -> dict:
    """Return default precision/recall dictionaries."""
    return {
        'precision_per_row': None, 'recall_per_row': None,
        'tp_per_row': None, 'fp_per_row': None, 'fn_per_row': None,
        'precision_per_rule_col': None, 'recall_per_rule_col': None,
        'tp_per_rule_col': None, 'fp_per_rule_col': None, 'fn_per_rule_col': None,
        'precision_per_col': None, 'recall_per_col': None,
        'tp_per_col': None, 'fp_per_col': None, 'fn_per_col': None
    }


def _get_default_precision_recall_per_rule_col() -> dict:
    """Return default precision/recall for per rule-col."""
    return {
        'precision_per_rule_col': None, 'recall_per_rule_col': None,
        'tp_per_rule_col': None, 'fp_per_rule_col': None, 'fn_per_rule_col': None
    }


def _get_default_precision_recall_per_col() -> dict:
    """Return default precision/recall for per col."""
    return {
        'precision_per_col': None, 'recall_per_col': None,
        'tp_per_col': None, 'fp_per_col': None, 'fn_per_col': None
    }


def get_precision_recall_from_validation(val_res_df: pd.DataFrame, all_col_arr: list) -> dict:
    """Calculate precision and recall from validation results."""
    true_err_col = val_res_df["err_col"].iloc[0] if "err_col" in val_res_df else []
    true_non_err_col = [col for col in all_col_arr if col not in true_err_col]
    
    tmp_df = val_res_df[val_res_df['wrong_vals'].notna()]
    pred_err_col = [x.replace("'", "").split(', ') for x in tmp_df[tmp_df['num_wrong'] > 0]['colname'].values]
    pred_err_col = list(itertools.chain(*pred_err_col))
    pred_non_err_col = [col for col in all_col_arr if col not in pred_err_col]
    
    precision, recall, tp, fp, fn = calculate_precision_recall(true_err_col, pred_err_col)
    
    return {
        'precision_per_row': precision, 'recall_per_row': recall,
        'tp_per_row': tp, 'fp_per_row': fp, 'fn_per_row': fn
    }


def get_precision_recall_from_validation_per_rule_col(val_res: list) -> dict:
    """Calculate precision and recall per rule-col from validation results."""
    err_col_idx_all = []
    wrong_idx_all = []
    
    for row in val_res:
        for i in range(len(row.get('colname', []))):
            colname_i = row['colname'][i]
            if isinstance(colname_i, str):
                suffix = '_' + colname_i.replace("'", "")
            else:
                suffix = '_' + str(colname_i).replace("'", "")
            
            err_idx = [str(e) + suffix for e in row.get('err_col_idx_true', [[]])[i]]
            err_col_idx_all.extend(err_idx)
            
            if len(row.get('wrong_idx', [None])) == 0 or row.get('wrong_idx', [None])[i] is None:
                new_list = []
            else:
                new_list = [str(e) + suffix for e in row.get('wrong_idx', [None])[i]]
            
            if len(new_list) != row.get('num_wrong', [0])[i]:
                if row.get('num_wrong', [0])[i] is not None:
                    new_list = [str(e) + suffix for e in range(0, row.get('num_wrong', [0])[i])]
            
            wrong_idx_all.extend(new_list if new_list else [])
    
    precision, recall, tp, fp, fn = calculate_precision_recall(err_col_idx_all, wrong_idx_all)
    
    return {
        'precision_per_rule_col': precision, 'recall_per_rule_col': recall,
        'tp_per_rule_col': tp, 'fp_per_rule_col': fp, 'fn_per_rule_col': fn
    }


def get_precision_recall_from_validation_per_col(val_res: list) -> dict:
    """Calculate precision and recall per col from validation results."""
    err_col_idx_all = []
    wrong_idx_all = []
    
    for row in val_res:
        for i in range(len(row.get('colname', []))):
            colname_i = row['colname'][i]
            if isinstance(colname_i, str):
                suffix = '_' + colname_i.replace("'", "")
            else:
                suffix = '_' + str(colname_i).replace("'", "")
            
            err_idx = [str(e) + suffix for e in row.get('err_col_idx_true', [[]])[i]]
            err_col_idx_all.extend(err_idx)
            
            if len(row.get('wrong_idx', [None])) == 0 or row.get('wrong_idx', [None])[i] is None:
                new_list = []
            else:
                new_list = [str(e) + suffix for e in row.get('wrong_idx', [None])[i]]
            
            if len(new_list) != row.get('num_wrong', [0])[i] and row.get('wrong_idx', [None])[i] is not None:
                new_list = [str(e) + suffix for e in range(0, row.get('num_wrong', [0])[i])]
            
            wrong_idx_all.extend(new_list if new_list else [])
    
    precision, recall, tp, fp, fn = calculate_precision_recall(err_col_idx_all, wrong_idx_all)
    
    return {
        'precision_per_col': precision, 'recall_per_col': recall,
        'tp_per_col': tp, 'fp_per_col': fp, 'fn_per_col': fn
    }


def _compare_positions(df: pd.DataFrame, col_target: str, rule_target: str) -> pd.DataFrame:
    """Compare positions of columns and rules."""
    def find_index_of_target(row: str, target: str) -> int:
        parts = row.split(';')
        try:
            return parts.index(f'{target}')
        except ValueError:
            return -1
    
    col_indices = df['errCol'].apply(lambda x: find_index_of_target(x, col_target))
    rule_indices = df['errRule'].apply(lambda x: find_index_of_target(x, rule_target))
    
    mask = col_indices == rule_indices
    return df[mask]
