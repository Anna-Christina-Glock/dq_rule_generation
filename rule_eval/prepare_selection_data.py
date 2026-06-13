"""
Prepare Selection Data Module for Rule Selection

This module provides utilities to convert comprehensive evaluation results
from `check_all_rules` into the specific DataFrame formats required by
the greedy rule selection algorithms (`df_pairs` and `df_rules`).
"""

import pandas as pd
from typing import List, Dict, Any, Optional


def prepare_selection_data_from_suite_li(
    suite_li: List[Dict[str, Any]],
    df_res_all: pd.DataFrame,
    use_only_relevant_rules: bool = False
) -> tuple:
    """
    Convert evaluation results to df_pairs and df_rules format.
    
    Args:
        suite_li: List of result dictionaries from check_all_rules
        df_res_all: Original dataframe with error specifications
        use_only_relevant_rules: If True, only include rules that passed precision check
        
    Returns:
        Tuple of (df_pairs, df_rules)
        
    df_pairs format:
        - ri: Rule index
        - pk: Primary key (row index)
        - col: Column(s) involved in the rule
        - isDirty: 1 if the row was actually dirty, 0 otherwise
        
    df_rules format:
        - ri: Rule index
        - col_set: Semicolon-separated list of columns
        - precision: Precision score for the rule
    """
    df_pairs_data = []
    df_rules_data = []
    
    for idx, result in enumerate(suite_li):
        ri = idx  # Rule index
        rule_count = result.get('rule_count', 0)
        exception_count = result.get('exception_count', 0)
        
        # Skip rules that couldn't run
        if rule_count == 0 and exception_count == 0:
            continue
            
        # Get precision from result
        precision = result.get('precision_per_rule_col', None)
        if precision is None:
            precision = 0.0
            
        # If filtering by precision, skip low-precision rules
        if use_only_relevant_rules and precision < 0.95:
            continue
        
        # Extract column set from validation results
        col_set = []
        val_res = result.get('val_res', None)
        if val_res is not None and len(val_res) > 0:
            col_set = val_res['colname'].iloc[0] if 'colname' in val_res.columns else []
            if isinstance(col_set, str):
                col_set = [col_set]
            elif isinstance(col_set, list):
                col_set = [str(c).replace("'", "") for c in col_set]
        else:
            # Try to get from col_set field
            col_set_str = result.get('col_set', result.get('colname', ''))
            if col_set_str:
                if isinstance(col_set_str, str):
                    col_set = [c.strip() for c in col_set_str.split(';') if c.strip()]
                elif isinstance(col_set_str, list):
                    col_set = [str(c).replace("'", "") for c in col_set_str]
        
        # Build column set string for df_rules
        col_set_str = ';'.join(col_set) if col_set else ''
        
        # Add to df_rules
        df_rules_data.append({
            'ri': ri,
            'col_set': col_set_str,
            'precision': precision
        })
        
        # Build df_pairs data from validation results
        # Look for unexpected indices and map them to columns
        if val_res is not None and len(val_res) > 0:
            # Process each expectation in the validation result
            for _, row in val_res.iterrows():
                colname = row.get('colname', '')
                if isinstance(colname, list):
                    colname = colname[0] if colname else ''
                    
                unexpected_indices = row.get('wrong_idx', [])
                if isinstance(unexpected_indices, str):
                    unexpected_indices = []
                    
                # Determine isDirty based on whether this row's pk is dirty
                # We need to find which rows this rule "caught"
                num_wrong = row.get('num_wrong', 0)
                num_wrong_true = row.get('num_wrong_true', 0)
                
                # If num_wrong > 0, this rule caught some dirty rows
                if num_wrong > 0 and num_wrong_true > 0:
                    # For each unexpected index, add a row to df_pairs
                    for pk in unexpected_indices:
                        df_pairs_data.append({
                            'ri': ri,
                            'pk': pk,
                            'col': colname if colname else col_set_str,
                            'isDirty': 1
                        })
                    
                    # If there are additional dirty rows not caught,
                    # we need to check the original df_res_all
                    pk_val = result.get('pk', None)
                    if pk_val is not None:
                        # Check if this specific pk is dirty
                        pk_mask = df_res_all['pk'] == pk_val
                        if pk_mask.any():
                            is_dirty = df_res_all.loc[pk_mask, 'isDirty'].iloc[0]
                            if is_dirty and len(unexpected_indices) == 0:
                                # No unexpected indices found, but pk is dirty
                                df_pairs_data.append({
                                    'ri': ri,
                                    'pk': pk_val,
                                    'col': colname if colname else col_set_str,
                                    'isDirty': 1
                                })
        
        # Also handle the isDirty=0 case (clean rows that shouldn't trigger)
        # This is harder to determine without the full context
        # We'll focus on the positive cases first
    
    # Create DataFrames
    df_pairs = pd.DataFrame(df_pairs_data, columns=['ri', 'pk', 'col', 'isDirty']) if df_pairs_data else pd.DataFrame(columns=['ri', 'pk', 'col', 'isDirty'])
    df_rules = pd.DataFrame(df_rules_data, columns=['ri', 'col_set', 'precision']) if df_rules_data else pd.DataFrame(columns=['ri', 'col_set', 'precision'])
    
    return df_pairs, df_rules


def prepare_selection_data_from_file(
    csv_path: str,
    df_res_all: pd.DataFrame,
    use_only_relevant_rules: bool = False
) -> tuple:
    """
    Load evaluation results from CSV file and convert to df_pairs and df_rules.
    
    Args:
        csv_path: Path to the evaluation results CSV
        df_res_all: Original dataframe with error specifications
        use_only_relevant_rules: If True, only include rules with precision >= 0.95
        
    Returns:
        Tuple of (df_pairs, df_rules)
    """
    # Load the CSV
    df_results = pd.read_csv(csv_path)
    
    # Convert to list of dicts (same format as suite_li)
    suite_li = df_results.to_dict('records')
    
    # Use the main function
    return prepare_selection_data_from_suite_li(suite_li, df_res_all, use_only_relevant_rules)