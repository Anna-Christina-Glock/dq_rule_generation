"""
Greedy Rule Selection Algorithms for Data Quality Rules

This module provides optimized greedy algorithms for selecting a minimal set of rules
that maximize coverage while maintaining a minimum precision threshold.
"""

import pandas as pd
from collections import defaultdict
from typing import Tuple, Dict, Set


def greedy_rule_selection_pairs(
    df_pairs: pd.DataFrame,
    df_rules: pd.DataFrame,
    precision_required: float = 1.0
) -> Tuple[list, dict, set]:
    """
    Greedy rule selection algorithm optimized for (row, column) pair coverage.
    
    This algorithm selects rules that cover the most uncovered (row, column) pairs
    while requiring a minimum precision threshold.
    
    Args:
        df_pairs: DataFrame with columns [ri, pk, col, isDirty]
        df_rules: DataFrame with columns [ri, col_set, precision, ...]
        precision_required: Minimum precision required for a rule to be considered (0.0 to 1.0)
        
    Returns:
        - selected_rules: List of rule IDs (ri) in selection order
        - coverage_by_rule: Dict mapping ri to coverage info with 'pairs' key
        - all_covered_pairs: Set of all (pk, col) tuples covered
    """
    # 1. Keep only rules with required precision
    good_rules = df_rules[df_rules["precision"] >= precision_required]["ri"]
    df = df_pairs[df_pairs["ri"].isin(good_rules) & (df_pairs["isDirty"] == 1)]
    
    # 2. Build rule_to_pairs using itertuples for ~5-10x speedup
    rule_to_pairs = {}
    for row in df.itertuples(index=False):
        ri = row.ri
        pk = row.pk
        col = row.col
        # Split col by semicolon if multiple columns
        for c in str(col).split(';'):
            c = c.strip()
            if ri not in rule_to_pairs:
                rule_to_pairs[ri] = set()
            rule_to_pairs[ri].add((pk, c))
    
    # 3. Greedy selection optimizing for (row, column) pairs
    selected = []
    covered_pairs = set()
    coverage_by_rule = {}
    
    remaining_rules = set(rule_to_pairs.keys())
    
    while remaining_rules:
        best_rule = None
        best_gain = 0
        
        for ri in remaining_rules:
            pairs = rule_to_pairs[ri]
            new_gain = len(pairs - covered_pairs)
            
            # Skip rules that add nothing
            if new_gain == 0:
                continue
            
            # Select rule with highest new (row, column) pair gain
            if new_gain > best_gain:
                best_rule = ri
                best_gain = new_gain
        
        # No rule adds new pairs
        if best_rule is None:
            break
        
        # Select the best rule
        selected.append(best_rule)
        new_pairs = rule_to_pairs[best_rule] - covered_pairs
        coverage_by_rule[best_rule] = {"pairs": new_pairs}
        
        # Update covered pairs set
        covered_pairs |= rule_to_pairs[best_rule]
        remaining_rules.remove(best_rule)
    
    return selected, coverage_by_rule, covered_pairs


def greedy_rule_selection(
    df_pairs: pd.DataFrame,
    df_rules: pd.DataFrame,
    precision_required: float = 1.0,
    row_weight: float = 1.0,
    col_weight: float = 0.5
) -> Tuple[list, dict, set, set]:
    """
    Greedy rule selection algorithm optimizing for combined row and column coverage.
    
    This algorithm selects rules that maximize a combined score of:
    - Row coverage (number of rows covered)
    - Column coverage (number of columns covered)
    While requiring a minimum precision threshold.
    
    Args:
        df_pairs: DataFrame with columns [ri, pk, isDirty]
        df_rules: DataFrame with columns [ri, col_set, precision, ...]
        precision_required: Minimum precision required for a rule to be considered (0.0 to 1.0)
        row_weight: Weight for row coverage in the combined score
        col_weight: Weight for column coverage in the combined score
        
    Returns:
        - selected_rules: List of rule IDs (ri) in selection order
        - coverage_by_rule: Dict mapping ri to coverage info with 'pks' and 'cols' keys
        - all_covered_pks: Set of all PKs covered by selected rules
        - all_covered_cols: Set of all columns covered by selected rules
    """
    import re
    
    # 1. Keep only rules with required precision
    good_rules = df_rules[df_rules["precision"] >= precision_required]["ri"]
    df = df_pairs[df_pairs["ri"].isin(good_rules) & (df_pairs["isDirty"] == 1)]
    
    # 2. Map: ri -> set(pk)
    rule_to_pks = (
        df.groupby("ri")["pk"].apply(lambda s: set(s)).to_dict()
    )
    
    # 3. Map: ri -> set of columns (extract from df_rules)
    rule_to_cols = {}
    for ri in rule_to_pks.keys():
        rule_row = df_rules[df_rules["ri"] == ri]
        if not rule_row.empty:
            col_set_str = rule_row.iloc[0]["col_set"]
            if pd.isna(col_set_str) or col_set_str == "":
                rule_to_cols[ri] = set()
            else:
                # Handle both single column and multiple columns (semicolon-separated)
                cols = [c.strip() for c in str(col_set_str).split(";") if c.strip()]
                rule_to_cols[ri] = set(cols)
    
    # 4. Greedy selection with row+column coverage
    selected = []
    covered_pks = set()
    covered_cols = set()
    coverage_by_rule = {}
    
    remaining_rules = set(rule_to_pks.keys())
    
    while remaining_rules:
        best_rule = None
        best_combined_gain = 0
        best_duplicates = None
        best_col_duplicates = None
        
        for ri in remaining_rules:
            pks = rule_to_pks[ri]
            cols = rule_to_cols.get(ri, set())
            
            new_row_gain = len(pks - covered_pks)
            new_col_gain = len(cols - covered_cols)
            duplicate_count = len(pks & covered_pks)
            col_duplicate_count = len(cols & covered_cols)
            
            # Skip rules that add nothing
            if new_row_gain == 0 and new_col_gain == 0:
                continue
            
            # Combined score: weighted sum of row and column gains
            combined_gain = (row_weight * new_row_gain) + (col_weight * new_col_gain)
            
            # Choose rule with:
            #   1) largest combined gain
            #   2) fewest row duplicates
            #   3) fewest column duplicates
            #   4) highest total coverage (rows)
            if (
                combined_gain > best_combined_gain
                or (combined_gain == best_combined_gain and duplicate_count < best_duplicates)
                or (combined_gain == best_combined_gain and duplicate_count == best_duplicates and col_duplicate_count < best_col_duplicates)
                or (
                    combined_gain == best_combined_gain 
                    and duplicate_count == best_duplicates 
                    and col_duplicate_count == best_col_duplicates
                    and len(pks) > len(rule_to_pks.get(best_rule, set()))
                )
            ):
                best_rule = ri
                best_combined_gain = combined_gain
                best_duplicates = duplicate_count
                best_col_duplicates = col_duplicate_count
        
        # No rule adds new coverage
        if best_rule is None:
            break
        
        # Select the best rule
        selected.append(best_rule)
        coverage_by_rule[best_rule] = {
            "pks": rule_to_pks[best_rule] - covered_pks,
            "cols": rule_to_cols[best_rule] - covered_cols
        }
        
        # Update covered PK and column sets
        covered_pks |= rule_to_pks[best_rule]
        covered_cols |= rule_to_cols[best_rule]
        remaining_rules.remove(best_rule)
    
    return selected, coverage_by_rule, covered_pks, covered_cols


def get_coverage_for_precision_levels(
    df_pairs: pd.DataFrame,
    df_rules: pd.DataFrame,
    precision_step: int = 5,
    method: str = "combined"
) -> list:
    """
    Calculate coverage at different precision thresholds.
    
    Args:
        df_pairs: DataFrame with columns [ri, pk, col, isDirty] (for pairs) or [ri, pk, isDirty] (for combined)
        df_rules: DataFrame with columns [ri, col_set, precision, ...]
        precision_step: Step size for precision thresholds (default: 5 for 0%, 5%, 10%, ...)
        method: Either "pairs" for pair-based coverage or "combined" for row+column coverage
        
    Returns:
        List of dictionaries with coverage information at each precision level
    """
    coverage_results = []
    
    for precValReq in range(0, 105, precision_step):
        precision_required = precValReq / 100.0
        
        if method == "pairs":
            selected_rules, coverage_by_rule, covered_pairs = greedy_rule_selection_pairs(
                df_pairs, df_rules, precision_required=precision_required
            )
            total_pairs = sum(len(c['pairs']) for c in coverage_by_rule.values())
            clean_count = len(df_pairs[(df_pairs["ri"].isin(selected_rules)) & (df_pairs["isDirty"] == 0)])
            
            coverage_results.append({
                'precValReq': round(precValReq / 10, 2),
                'total_pairs': total_pairs,
                'clean_val': clean_count,
                'covered_pairs': covered_pairs,
                'selected_rules': selected_rules,
                'coverage_by_rule': coverage_by_rule
            })
            
        else:  # combined
            selected_rules, coverage_by_rule, covered_pks, covered_cols = greedy_rule_selection(
                df_pairs, df_rules, precision_required=precision_required
            )
            clean_count = len(df_pairs[(df_pairs["ri"].isin(selected_rules)) & (df_pairs["isDirty"] == 0)])
            
            coverage_results.append({
                'precValReq': round(precValReq / 10, 2),
                'total_coverage': len(covered_pks),
                'clean_val': clean_count,
                'covered_pks': covered_pks,
                'covered_cols': covered_cols,
                'selected_rules': selected_rules,
                'coverage_by_rule': coverage_by_rule
            })
    
    return coverage_results