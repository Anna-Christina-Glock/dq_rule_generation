"""Evaluate saved GX rule results and export rule-coverage summaries.

The default mode evaluates one dataset and writes the legacy metric, SQLite,
and coverage outputs. Set both REDUCTION_DATA_FILE and COVERAGE_DATA_FILE to
run paired A/B coverage instead. The runner reads PATTERN_NAME, DATA_IN,
DATA_OUT, RULE_RESULTS_PATH, and the optional PATTERN_LI and USE_GLOB
environment variables; paths are intended for the Docker mount layout.
"""
import os
import csv
from pathlib import Path
import great_expectations as gx
from great_expectations.expectations import *
import re
import json
import time
from os import listdir
from os.path import isfile, join
import math
import pandas as pd
from collections import defaultdict
import pickle
import sqlite3
import numpy as np
from glob import glob



def greedy_rule_selection_pairs(df_pairs, df_rules, precision_required=1):
    """
    df_pairs:   DataFrame with columns [ri, pk, col, isDirty]
    df_rules:   DataFrame with columns [ri, col_set, precision, ...]
    
    Returns: 
      - selected_rules: list of ri
      - coverage_by_rule: dict[ri] -> dict with 'pairs' (set of (pk, col) tuples)
      - all_covered_pairs: set of all (pk, col) pairs covered
    """
    # 1. Keep only rules with required precision
    good_rules = df_rules[df_rules["precision"] >= precision_required]["ri"]
    df = df_pairs[df_pairs["ri"].isin(good_rules) & (df_pairs["isDirty"] == 1)]

    # Preserve actual (row, column) pairs instead of inferring a Cartesian product.
    # itertuples avoids constructing a pandas Series for every row.
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

    # 5. Greedy selection optimizing for (row, column) pairs
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


def greedy_rule_selection(df_pairs, df_rules, precision_required=1, row_weight=1.0, col_weight=0.5):
    """
    df_pairs:   DataFrame with columns [ri, pk, isDirty]
    df_rules:   DataFrame with columns [ri, col_set, precision, ...]
    
    Returns: 
      - selected_rules: list of ri
      - coverage_by_rule: dict[ri] -> newly added PKs
      - all_covered_pks: set of all PKs covered by selected rules
      - all_covered_cols: set of all columns covered by selected rules
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


def getErrCount(conn, cur, precVal, errorCount, indexVec, dataDf):
    # Precompute index lookup
    index_map = {v: i for i, v in enumerate(indexVec)}
    dirty_idx = index_map[-99]
    
    # Precompute column lookup
    if 'beer-name' in dataDf.columns:
        col_index = {col: i for i, col in enumerate(dataDf.columns)}
    else:
        col_index = {col: i+1 for i, col in enumerate(dataDf.columns)}

    # Fetch rule info
    cur.execute(f"SELECT * FROM ruleInfo WHERE precision >= {precVal}")
    rule_info = cur.fetchall()

    # Fetch all ruleAndPK rows at once
    query = f"""
    SELECT ruleAndPK.*
    FROM ruleAndPK
    JOIN ruleInfo ON ruleAndPK.ri = ruleInfo.ri
    WHERE ruleInfo.precision >= {precVal}
    """
    cur.execute(query)
    rule_pk = cur.fetchall()

    # Group ruleAndPK rows by rule id (now has 4 cols: pk, ri, col, isDirty)
    rules = defaultdict(list)
    for pk, ri, col, flag in rule_pk:
        rules[ri].append((pk, flag, ri))

    # Process rules
    for _, ri, col_name, _, _ in rule_info:
        if ';' in col_name:
            errColIdx = []
            for strPart in col_name.split(';'):
                errColIdx.append(col_index[strPart])
        else:
            errColIdx = col_index[col_name]

        rows = rules[ri]

        # Apply each rule's dirty and clean counts to its target cells.
        rows_1 = [pk for pk, flag, _ in rows if flag == 1]
        rows_0_count = sum(flag != 1 for _, flag, _ in rows)

        if rows_1:
            errorCount[[index_map[r] for r in rows_1], errColIdx] += 1

        if rows_0_count:
            errorCount[dirty_idx, errColIdx] += rows_0_count
    return errorCount


def getCoverageLi_pairs(df_pairs, df_rules):
    coverageLi = []
    for precValReq in range(0, 105, 5):
        selected_rules, coverage_by_rule, covered_pairs = greedy_rule_selection_pairs(df_pairs, df_rules, precision_required=precValReq*(1/100))
        df_clean = df_pairs[df_pairs["ri"].isin(selected_rules) & (df_pairs["isDirty"] == 0)]
        df_clean_info = df_rules[df_rules['ri'].isin(selected_rules)].loc[:, ['ri', 'col_set']]
        df_clean_merge = pd.merge(df_clean_info, df_clean, 'inner', on='ri')
        setClean = df_clean_merge.loc[:, ['pk', 'col_set']].drop_duplicates() 

        # Count total (row, column) pairs covered
        total_pairs = sum(len(c['pairs']) for c in coverage_by_rule.values())
        
        print(f"\t{precValReq*(1/10):.2f} -> Total (row,col) pairs coverage: {total_pairs}/{len(setClean)}")
        print(f"\t\tSelected rules in order({len(selected_rules)}):")
        coverageLi.append({'precValReq': round(precValReq*(1/10), 2), 'total_pairs': total_pairs, 'clean_val': len(setClean),
                        'covered_pairs': covered_pairs, 'selected_rules': selected_rules, 'coverage_by_rule': coverage_by_rule})
    return coverageLi


def getCoverageLi(df_pairs, df_rules):
    coverageLi = []
    for precValReq in range(0, 105, 5):
        selected_rules, coverage_by_rule, covered_pks, covered_cols = greedy_rule_selection(df_pairs, df_rules, precision_required=precValReq*(1/100))
        df_clean = df_pairs[df_pairs["ri"].isin(selected_rules) & (df_pairs["isDirty"] == 0)]
        df_clean_info = df_rules[df_rules['ri'].isin(selected_rules)].loc[:, ['ri', 'col_set']]
        df_clean_merge = pd.merge(df_clean_info, df_clean, 'inner', on='ri')
        setClean = df_clean_merge.loc[:, ['pk', 'col_set']].drop_duplicates() 

        print(f"\t{precValReq*(1/10):.2f} -> Total coverage: {len(set(covered_pks))}/{len(setClean)}")
        print(f"\t\tColumns covered: {len(covered_cols)}")
        print(f"\t\tSelected rules in order({len(selected_rules)}):")
        coverageLi.append({'precValReq': round(precValReq*(1/10), 2), 'total_coverage': len(set(covered_pks)), 'clean_val': len(setClean),
                        'covered_pks': covered_pks, 'covered_cols': covered_cols, 'selected_rules': selected_rules, 'coverage_by_rule': coverage_by_rule})
    return coverageLi


def getData(directory, fname):
    dataDf = pd.read_csv(Path(directory)/fname)
    return dataDf


def collapseDataframe(arr):

    data = arr[:-1]      # all except last row
    last_row = arr[-1:]  # keep the last row unchanged

    colLen = arr.shape[1]
    # Extract grouping columns
    g1 = data[:, colLen-2].astype(str)
    g2 = data[:, colLen-1].astype(str)

    # Combine them into a single unique key
    group_keys = np.char.add(np.char.add(g1, np.full(len(g1), "|", dtype='U')), g2)

    # Identify unique groups
    unique_keys, inverse_idx = np.unique(group_keys, return_inverse=True)

    num_groups = len(unique_keys)
    # Allocate an object array for grouped totals and the preserved final row.
    result = np.empty((num_groups + 1, colLen), dtype=object)

    # Aggregate each non-grouping column across rows with matching labels.
    for col in range(colLen-2):
        # Convert numeric text to floats; treat other values as zero.
        col_values = data[:, col]
        try:
            numeric = col_values.astype(float)
        except:
            numeric = np.array([float(x) if str(x).replace('.','',1).isdigit() else 0.0
                                for x in col_values])
        
        result[:num_groups, col] = np.bincount(
            inverse_idx,
            weights=numeric,
            minlength=num_groups
        )

    # --- Split grouping keys back into strings ---
    split = np.array([k.split("|") for k in unique_keys], dtype=object)

    result[:num_groups, colLen-2] = split[:, 0]
    result[:num_groups, colLen-1] = split[:, 1]

    # Preserve the input's final row without numeric conversion.
    result[num_groups, :] = last_row


    # Append the last row unchanged
    result = np.vstack([result, last_row])
    return result


def parseErrColVal(errColVal):
    """Parse errCol's semicolon- and <->-separated exact column names."""
    if errColVal is None or (isinstance(errColVal, float) and math.isnan(errColVal)):
        return set()
    if not isinstance(errColVal, str):
        return set()

    return {
        column.strip()
        for group in errColVal.split(';')
        for column in group.split('<->')
        if column.strip()
    }


def checkNameInErrcol(errColVal, colName):
    return colName in parseErrColVal(errColVal)


def getErrInfoDf(dataDf):
    data_columns = list(dataDf.columns[:14])
    indices_by_col = {colName: [] for colName in data_columns}
    for row_index, errColVal in dataDf['errCol'].items():
        for colName in parseErrColVal(errColVal):
            if colName in indices_by_col:
                indices_by_col[colName].append(row_index)

    errFilLi = [
        {
            'colName': colName,
            'len': len(indices_by_col[colName]),
            'index': pd.Index(indices_by_col[colName]),
        }
        for colName in data_columns
    ]

    errInfoDf = pd.DataFrame(errFilLi)
    return errInfoDf


def getMaInfo(dataDf):
    context, batch = getBatch(dataDf, data_source_name='ma_test_1')
    suite = gx.ExpectationSuite(name="missing_suite", meta={"created_by": "data_team"})
    for colName in dataDf.columns[0:14]:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=colName))
    results = batch.validate(suite, result_format="COMPLETE")

    # Extract missing row indices
    maLi = list()
    for result in results.results:
        maCount = result.result.get('unexpected_count')
        maLi.append({'colname': result.expectation_config.kwargs.get('column'), 'unexpected_count': maCount,
                     'unexpected_index_list': result.result['unexpected_index_list']})

    maDf = pd.DataFrame(maLi)
    return(maDf)


def getAllInfoDfs(directory, fname, progress_callback=None):
    if progress_callback:
        progress_callback(f"reading CSV {fname}")
    dataDf = getData(directory, fname)
    required_columns = {"isDirty", "errCol"}
    missing_columns = required_columns - set(dataDf.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if progress_callback:
        progress_callback(f"building dirty-cell metadata for {len(dataDf):,} rows")
    errInfoDf = getErrInfoDf(dataDf)
    if progress_callback:
        progress_callback("running Great Expectations missing-value validation")
    maDf = getMaInfo(dataDf)
    if progress_callback:
        progress_callback("finished missing-value validation")
    return errInfoDf, maDf, dataDf


def getFileLi(mypath, pattern, useGlob=False):
    if not useGlob:
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and pattern in f]
    else:
        onlyfiles = glob(os.path.join(mypath, pattern))

    for f in onlyfiles:
        if '_Size' in f:
            continue
    return onlyfiles


def getBatch(df, data_source_name="pci_parts", asset_name="pci_parts_asset", batch_definition_name="pci_parts_batch"):
    context = gx.get_context(mode="ephemeral")
    ds_names = set(context.data_sources.all())
    if data_source_name in ds_names:
        data_source = context.data_sources.get(data_source_name)
    else:
        data_source = context.data_sources.add_pandas(name=data_source_name)

    assetNames = list()
    for dataAss in data_source.assets:
        assetNames.append(dataAss.name)
    if asset_name in assetNames:
        asset = data_source.get_asset(asset_name)
    else:
        asset = data_source.add_dataframe_asset(name=asset_name)

    try:
        batch_def = asset.get_batch_definition(batch_definition_name)
    except Exception:
        batch_def = asset.add_batch_definition_whole_dataframe(batch_definition_name)

    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    return context, batch


def calculate_precision_recall(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    laplas_precision = (TP + 1) / (TP + FP + 2) if (TP + FP + 2) != 0 else 0
    laplas_recall = (TP + 1) / (TP + FN + 2) if (TP + FN + 2) != 0 else 0

    return precision, recall, laplas_precision, laplas_recall


def getMetrics_fast2(res, errInfoDf, maDf, ri, dataDf, err_index_by_col=None, ma_index_by_col=None, isVerbose=False):
    colLi = [res['expectation_config']['kwargs'][key] 
             for key in res['expectation_config']['kwargs'] 
             if 'column' in key]
    
    if len(colLi) > 0 and isinstance(colLi[0], list):
        colLi = [item for sublist in colLi for item in sublist]

    if len(colLi) == 0:
        col_set = []
    else:
        if isinstance(colLi[0], dict):
            col_set = [col.replace('stra\\u00dfe', 'straße') for col in colLi[0].keys()]
        else:
            col_set = [str(col).replace('stra\\u00dfe', 'straße') for col in set(colLi)]
    
    dirty_by_col = {col: set() for col in col_set}
    if err_index_by_col is not None:
        for col in col_set:
            if col in err_index_by_col:
                dirty_by_col[col] = err_index_by_col[col]
    else:
        for _, row in errInfoDf.iterrows():
            if row['colName'] in col_set:
                dirty_by_col[row['colName']].update(row['index'])
    
    set1 = set()
    setLi = list()
    isSetZero = False
    if ma_index_by_col is not None:
        for col in col_set:
            if col in ma_index_by_col:
                setLi.append(ma_index_by_col[col])
                if len(set1) == 0:
                    set1.update(ma_index_by_col[col])
                else:
                    set2 = set1 & ma_index_by_col[col]
                    if len(set2) == 0:
                        set1.update(ma_index_by_col[col])
                        isSetZero = True
                    else:
                        set1 = set2
                        isSetZero = False
    else:
        for _, row in maDf.iterrows():
            if row['colname'] in col_set:
                setLi.append(set(row['unexpected_index_list']))
                if len(set1) == 0:
                    set1.update(row['unexpected_index_list'])
                else:
                    set2 = set1 & set(row['unexpected_index_list'])
                    if len(set2) == 0:
                        set1.update(row['unexpected_index_list'])
                        isSetZero = True
                    else:
                        set1 = set2
                        isSetZero = False
    if isSetZero:
        maCount = 0
    else:
        maCount = len(set1)
    
    if 'unexpected_index_list' in list(res['result'].keys()):
        unexpected_indices = set(res['result']['unexpected_index_list'])
    else:
        value_set = res['expectation_config']['kwargs'].get("value_set", [])
        observed_values = res['result'].get("observed_value") or []
        if isinstance(observed_values, float):
            observed_values = [observed_values]
        if isinstance(observed_values, int):
            observed_values = [observed_values]
        value_counts = res['result'].get("details", {}).get("value_counts", {})
        
        unexpected_distinct_values = [v for v in observed_values if v not in value_set]

        total_incorrect_rows = 0
        for item in value_counts:
            val = item.get("value")
            count = item.get("count", 0)
            
            if val in unexpected_distinct_values:
                total_incorrect_rows += count
        unexpected_indices = total_incorrect_rows
        col_name = res['expectation_config']['kwargs'].get('column')
        
        if col_name and col_name in dataDf.columns:
            mask = dataDf[col_name].isin(unexpected_distinct_values)
            unexpected_indices = set(list(dataDf[mask].index))
        else:
            unexpected_indices = set()
    
    dirty_set = set()
    for col, indices in dirty_by_col.items():
        dirty_set.update(indices)
    
    tp = len(dirty_set & unexpected_indices) + maCount
    fp = len(unexpected_indices - dirty_set)
    fn = len(dirty_set - unexpected_indices)

    precision, recall, laplas_precision, laplas_recall = calculate_precision_recall(tp, fp, fn)
    if isVerbose:
        print(f"Columns: {colLi}")
        print(f"Dirty indices: {len(dirty_set)}")
        print(f"Unexpected indices: {len(unexpected_indices)}")
        print(f"True positives: {tp}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")

    ruleType = res['expectation_config']['type']
    dirty_by_col_set = {col: dirty_by_col.get(col, set()) for col in col_set}
    
    riAndPKData = []
    for pk in unexpected_indices:
        pk_cols = []
        for col in col_set:
            if pk in dirty_by_col.get(col, set()):
                pk_cols.append(col)
        
        if pk_cols:
            isDirty = 1
            col_str = ';'.join(pk_cols)
        else:
            isDirty = 0
            col_str = ';'.join(col_set) if col_set else ''
        
        riAndPKData.append((pk, ri, col_str, isDirty))

    return tp, fp, fn, precision, recall, laplas_precision, laplas_recall, col_set, unexpected_indices, ruleType, riAndPKData


def createDescStr(typeStr, kwargsDict):
    keylist = [key for key in kwargsDict.keys()]
    match typeStr:
        case 'expect_column_value_lengths_to_be_between':
            descStr = f"'{kwargsDict['column']}' length to be between {kwargsDict.get('min_value',0)} and {kwargsDict.get('max_value',-99)}"
        case 'expect_column_values_to_be_of_type':
            descStr = f"'{kwargsDict['column']}' to be of type {kwargsDict['type_']}"
        case 'expect_column_values_to_be_in_set':
            descStr = f"'{kwargsDict['column']}' to be in set {kwargsDict['value_set']}"
        case 'expect_column_values_to_not_match_regex':
            descStr = f"'{kwargsDict['column']}' to not match regex '{kwargsDict['regex']}'"
        case 'expect_column_values_to_be_between':
            if 'max_value' in keylist:
                descStr = f"'{kwargsDict['column']}' value to be between {kwargsDict.get('min_value',-99)} and {kwargsDict.get('max_value',99)}"
            else:
                descStr = f"'{kwargsDict['column']}' value to be between {kwargsDict['min_value']}"
        case 'expect_column_value_lengths_to_equal':
            descStr = f"'{kwargsDict['column']}' length to be equal {kwargsDict['value']}"
        case 'expect_column_values_to_not_match_regex_list':
            descStr = f"'{kwargsDict['column']}' to match not regex list {kwargsDict['regex_list']}"
        case 'expect_column_values_to_be_null':
            descStr = f"'{kwargsDict['column']}' to be null"
        case 'expect_column_values_to_be_dateutil_parseable':
            descStr = f"'{kwargsDict['column']}' to be dateutil parseable"
        case 'expect_column_pair_values_a_to_be_greater_than_b':
            descStr = f"'{kwargsDict['column_A']}' to be greater than '{kwargsDict['column_B']}'"
        case 'expect_column_values_to_not_be_in_set':
            descStr = f"'{kwargsDict['column']}' to be not in set {kwargsDict['value_set']}"
        case 'expect_column_values_to_match_regex_list':
            descStr = f"'{kwargsDict['column']}' to match regex list {kwargsDict['regex_list']}"
        case 'expect_column_values_to_not_be_null':
            descStr = f"'{kwargsDict['column']}' to be not null"
        case 'expect_column_pair_values_to_be_equal':
            descStr = f"'{kwargsDict['column_A']}' to be equal to '{kwargsDict['column_B']}'"
        case 'expect_column_values_to_match_strftime_format':
            descStr = f"'{kwargsDict['column']}' to match strftime format {kwargsDict['strftime_format']}"
        case 'expect_column_values_to_be_in_type_list':
            if 'type_list' in keylist:
                descStr = f"'{kwargsDict['column']}' to be in type list {kwargsDict['type_list']}"
            elif 'value_stype_listet' in keylist:
                descStr = f"'{kwargsDict['column']}' to be in type list {kwargsDict['value_stype_listet']}"
            else:
                descStr = f"'{kwargsDict['column']}' value to be between {kwargsDict}"
        case 'expect_column_values_to_match_regex':
            if 'regex' in kwargsDict.keys() and 'column' in kwargsDict.keys():
                descStr = f"'{kwargsDict['column']}' to match regex '{kwargsDict['regex']}'"
            else:
                descStr = ''
        case 'expect_column_values_to_not_match_like_pattern':
            descStr = f"'{kwargsDict['column']}' to match like pattern '{kwargsDict['like_pattern']}'"
        case _:
            descStr = ''
            print(typeStr)
    return descStr


def getFormattedTime(timeInfo):
    text = timeInfo
    pattern = r"=\s[0-9.]*"
    matches = re.findall(pattern, text)
    if len(matches)>0:
        timeStr = matches[0].replace('= ','')
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(float(timeStr)))
    else:
        formatted_time = None
    return formatted_time


def getFileInfo(f):
    suiteStr = None
    winStr = None
    suiteMatch = re.findall(r"Suite[0-9]*", f)
    if len(suiteMatch)>0:
        suiteStr = suiteMatch[0].replace('Suite','')
    winMatch = re.findall(r"Win[0-9]*", f)
    if len(winMatch)>0:
        winStr = winMatch[0].replace('Win','')
    return suiteStr, winStr


def replaceProblematicPattern(lineStrSplit, patternMatch=r".{,5}[^\\]\\\\\\\\\\[^\\].{,5}", patternFind="\\\\\\\\\\", patternRep=""):
    allMatches = re.findall(patternMatch, lineStrSplit)
    for matchStr in allMatches:
        lineStrSplit = lineStrSplit.replace(matchStr, matchStr.replace(patternFind, patternRep))
    return lineStrSplit


def preprocessLineStr(lineStr):
    splitPattern = "validation_result:"
    timeInfo = lineStr.split(splitPattern)[0]
    timeInfo = getFormattedTime(timeInfo)
    lineStrSplit = lineStr.split(splitPattern)[1].replace('\\\\n', '??????').replace('\\\\\\"', '+üöäü+')
    lineStrSplit = lineStrSplit[2:(len(lineStrSplit)-1)].strip().replace('\ufeff', '').replace('\r', '').replace('\t', ' ')
    lineStrSplit = lineStrSplit.replace('\\n', '\n').replace('\\"', '"').replace('\\', '\\\\')
    lineStrSplit = lineStrSplit.replace('??????', '\\\\n').replace('+üöäü+', '\\"')
    
    lineStrSplit = replaceProblematicPattern(lineStrSplit, r".{,5}[^\\]\\\\\\\\\\[^\\].{,5}", "\\\\\\\\\\\\\\", "")
    lineStrSplit = replaceProblematicPattern(lineStrSplit, r"\"regex\": \"\\\\\\\\\\\\\\\\\\\\\\\\\\\"",
                                             "\"regex\": \"\\\\\\\\\\\\\\\\\\\\\\\\\\\"", "\"regex\": \"backslashes(removed for parsing)\"")
    lineStrSplit = replaceProblematicPattern(lineStrSplit, r"\\\"U\\\\\\\\\\\\\\\\\\\\\\\\\\\"",
                                             "\\\\\\\\\\\\\\\\\\\\\\\\\\",
                                             "backslashes(removed for parsing)")
    lineStrSplit = replaceProblematicPattern(lineStrSplit, r"9/\\\\\\3",
                                             "\\\\\\", "backslashes(removed for parsing)")
    lineStrSplit = replaceProblematicPattern(lineStrSplit, r"52}\\\\\\3",
                                             "\\\\\\", "backslashes(removed for parsing)")
    lineStrSplit = replaceProblematicPattern(lineStrSplit, r"\"\\\\\\4",
                                             "\\\\\\", "backslashes(removed for parsing)")
    lineStrSplit = replaceProblematicPattern(lineStrSplit, r"\"\\\\\\2",
                                             "\\\\\\", "backslashes(removed for parsing)")
    lineStrSplit = replaceProblematicPattern(lineStrSplit, r"\"regex\": \"\^\\\\\\\\\\\\\\\\\\\\\\\\\\\"",
                                             "\\\\\\\\\\\\\\\\\\\\\\\\\\", "backslashes(removed for parsing)")
    
    return timeInfo, lineStrSplit


def countRules(jsonStr):        
    successRules = 0
    errRules = 0
    allRules = 0
    for res in jsonStr['results']:
        if len(res['result']) > 0:
            successRules += 1
        if 'raised_exception' not in res['exception_info'].keys():
            errRules += 1
        allRules += 1
    return successRules, errRules, allRules


def createDB(dir, pattern):
    dbName = f"groups_all_1_{pattern}_1.db"
    conn = sqlite3.connect(Path(dir)/dbName)
    cur = conn.cursor()

    cur.execute("""DROP TABLE IF EXISTS ruleInfo;""")
    cur.execute("""DROP TABLE IF EXISTS ruleAndPK;""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ruleInfo (
        ruleType INTEGER NOT NULL,
        ri INTEGER NOT NULL,
        col_set INTEGER NOT NULL,
        precision INTEGER NOT NULL,
        recall INTEGER NOT NULL
    )
    """)    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS foundRows (
        errType Varchar NOT NULL,
        errCol Varchar NOT NULL,
        foundErrs INTEGER NOT NULL
    )
    """)  
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ruleAndPK (
        pk integer NOT NULL,
        ri integer NOT NULL,
        col TEXT NOT NULL,
        isDirty integer NOT NULL
    )
    """)
    
    cur.execute("PRAGMA journal_mode = MEMORY;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA cache_size = -1000000;")

    return conn, cur, dbName


def _get_pattern_glob_groups(pattern_name):
    """Build the original ordered glob patterns, grouped by precision setting."""
    settings = ["9_0", "10_0", "8_0", "10_2", "8_2", "9_2", "11_0", "11_2"]
    if pattern_name == "pci":
        return [
            (
                setting,
                f"resDf_all__*{setting}_*_new_OnlyRelevanRules_valChange_newValInfo_blub0_"
                f"BigName_trainGenData_adressPerson_*_ValiBigOut_*.txt",
            )
            for setting in settings
        ]
    if pattern_name in {"hospital", "flights", "beers", "tax", "tax_gemma4", "tax_glm"}:
        if pattern_name == "tax_gemma4" or "tax_glm":
            bigName_pattern = f"BigName_tax_*_ValiBigOut_*.txt"
        else:
            bigName_pattern = f"BigName_{pattern_name}_*_ValiBigOut_*.txt"

        return [
            (
                setting,
                f"resDf_all__*{setting}_*_new_{pattern_name}_OnlyRelevanRules_valChange_newValInfo_blub0_"
                f"{bigName_pattern}",
            )
            for setting in settings
        ]

    raise ValueError(f"Unsupported PATTERN_NAME: {pattern_name}")


def _get_paired_pattern_groups(pattern_name, dataset_stem):
    """Use the original glob groups; dataset markers later select A or B files."""
    use_glob = os.environ.get("USE_GLOB", "True").lower() == "true"
    if pattern_name == "tax" or pattern_name == "tax_gemma4" or pattern_name == "tax_glm":
        use_glob = True

    pattern_li_override = os.environ.get("PATTERN_LI")
    if not use_glob:
        if not pattern_li_override:
            raise ValueError("Paired mode with USE_GLOB=False requires PATTERN_LI")
        import ast
        pattern_li = ast.literal_eval(pattern_li_override)
        groups = []
        for i, pattern in enumerate(pattern_li):
            pattern = str(pattern)
            pattern = re.sub(r"(?<=BigName_).*?(?=_ValiBigOut_)", dataset_stem, pattern)
            groups.append((f"pattern_{i + 1}", [pattern], False))
        return groups

    print(f"------- pattern_name: {pattern_name} -------")
    return [
        (group_name, [pattern], True)
        for group_name, pattern in _get_pattern_glob_groups(pattern_name)
    ]


def _paired_log(message):
    print(f"[paired] {message}", flush=True)


def _prepare_paired_dataset(data_file, data_directory, dataset_role):
    """Load one A/B dataset and precompute the same validation metadata as the original path."""
    load_started = time.perf_counter()
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = Path(data_directory) / data_path
    if data_path.suffix.lower() != ".csv":
        data_path = data_path.with_suffix(".csv")
    if not data_path.exists():
        raise FileNotFoundError(f"{dataset_role} dataset not found: {data_path}")

    _paired_log(f"Loading {dataset_role} dataset: {data_path}")
    err_info_df, ma_df, data_df = getAllInfoDfs(
        data_path.parent,
        data_path.name,
        progress_callback=lambda stage: _paired_log(f"{dataset_role} dataset: {stage}"),
    )
    err_index_by_col = {
        row.colName: set(row.index)
        for row in err_info_df.itertuples(index=False)
    }
    ma_index_by_col = {
        row.colname: set(row.unexpected_index_list)
        for row in ma_df.itertuples(index=False)
    }
    dirty_row_ids = set(data_df.index[data_df["isDirty"].astype(bool)])
    dirty_cell_pairs = set()
    for row in err_info_df.itertuples(index=False):
        dirty_cell_pairs.update(
            (index, row.colName) for index in row.index if index in dirty_row_ids
        )
    _paired_log(
        f"Loaded {dataset_role} dataset: {len(data_df):,} rows, "
        f"{len(dirty_row_ids):,} dirty rows, {len(dirty_cell_pairs):,} dirty row/column pairs "
        f"({time.perf_counter() - load_started:.1f}s)"
    )

    return {
        "data_path": data_path,
        "data_df": data_df,
        "err_info_df": err_info_df,
        "ma_df": ma_df,
        "err_index_by_col": err_index_by_col,
        "ma_index_by_col": ma_index_by_col,
        "dirty_row_ids": dirty_row_ids,
        "dirty_cell_pairs": dirty_cell_pairs,
    }


def _load_paired_rule_results(dataset, results_directory, dataset_role,
                              result_patterns, use_glob, group_name):
    """Evaluate matching rule-result files against one already-loaded dataset."""
    load_started = time.perf_counter()
    data_path = dataset["data_path"]
    data_df = dataset["data_df"]
    err_info_df = dataset["err_info_df"]
    ma_df = dataset["ma_df"]
    err_index_by_col = dataset["err_index_by_col"]
    ma_index_by_col = dataset["ma_index_by_col"]
    dirty_row_ids = dataset["dirty_row_ids"]
    dirty_cell_pairs = dataset["dirty_cell_pairs"]

    dataset_stem = data_path.stem
    marker = f"BigName_{dataset_stem}_ValiBigOut_"
    results_path = Path(results_directory)
    result_files = set()
    for pattern in result_patterns:
        if use_glob:
            result_files.update(results_path.glob(pattern))
        else:
            result_files.update(
                path for path in results_path.iterdir()
                if path.is_file() and pattern in path.name
            )
    result_files = sorted(result_files)
    result_files = [path for path in result_files if marker in path.name]
    result_files = [path for path in result_files if re.search(r"_Size\d+\.txt$", path.name)]
    _paired_log(
        f"Group {group_name}, {dataset_role} dataset: found {len(result_files):,} matching result files"
    )
    if not result_files:
        raise FileNotFoundError(
            f"No validation result files for {dataset_role} dataset '{dataset_stem}' in {results_directory}. "
            f"Expected files matching {result_patterns} and containing {marker}"
        )

    rule_metrics = {}
    rule_pairs = []
    for file_index, result_file in enumerate(result_files, start=1):
        if file_index == 1 or file_index % 50 == 0 or file_index == len(result_files):
            _paired_log(
                f"Group {group_name}, {dataset_role}: reading result file "
                f"{file_index:,}/{len(result_files):,} ({result_file.name})"
            )
        normalized_file = result_file.name.replace(marker, "BigName_<DATASET>_ValiBigOut_", 1)
        source_run = normalized_file.split("_BigName_<DATASET>_ValiBigOut_", 1)[0]

        with result_file.open("r", encoding="utf-8") as handle:
            encoded_result = json.load(handle)
        if not isinstance(encoded_result, str) or not encoded_result.strip() or encoded_result.strip() == "[]":
            continue
        try:
            validation_result = json.loads(encoded_result)
        except (json.JSONDecodeError, TypeError):
            print(f"Skipping invalid validation result: {result_file}")
            continue

        for result in validation_result.get("results", []):
            exception_info = result.get("exception_info", {})
            if "raised_exception" not in exception_info or exception_info["raised_exception"]:
                continue

            expectation_config = result.get("expectation_config", {})
            # Suite and window IDs describe how results were partitioned for
            # validation. They can differ between A and B even when the rule
            # generation run and expectation are identical, so they must not
            # participate in the paired A/B identity.
            rule_identity = json.dumps(
                [source_run, expectation_config],
                sort_keys=True,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
            metrics = getMetrics_fast2(
                result,
                err_info_df,
                ma_df,
                rule_identity,
                data_df,
                err_index_by_col=err_index_by_col,
                ma_index_by_col=ma_index_by_col,
            )
            rule_metrics[rule_identity] = {
                "ri": rule_identity,
                "col_set": ";".join(metrics[7]),
                "precision": metrics[3],
                "recall": metrics[4],
            }
            rule_pairs.extend(metrics[10])

    if not rule_metrics:
        _paired_log(f"Group {group_name}, {dataset_role}: no usable rules found")
        raise ValueError(f"No usable rules found for {dataset_role} dataset '{dataset_stem}'")
    _paired_log(
        f"Group {group_name}, {dataset_role}: parsed {len(result_files):,} files, "
        f"{len(rule_metrics):,} usable rules, {len(rule_pairs):,} rule/row records "
        f"({time.perf_counter() - load_started:.1f}s)"
    )
    return (
        data_df,
        pd.DataFrame(rule_metrics.values()),
        pd.DataFrame(rule_pairs, columns=["pk", "ri", "col", "isDirty"]),
        dirty_row_ids,
        dirty_cell_pairs,
    )


def _make_paired_coverage_rows(reduction_rules, reduction_pairs, coverage_rules, coverage_pairs,
                               group_name):
    shared_rules = set(reduction_rules["ri"]) & set(coverage_rules["ri"])
    _paired_log(
        f"Group {group_name}: {reduction_rules['ri'].nunique():,} A rules, "
        f"{coverage_rules['ri'].nunique():,} B rules, "
        f"{len(shared_rules):,} shared rule definitions"
    )
    if not shared_rules:
        return None
    reduction_rules = reduction_rules[reduction_rules["ri"].isin(shared_rules)].copy()
    coverage_rules = coverage_rules[coverage_rules["ri"].isin(shared_rules)].copy()
    reduction_pairs = reduction_pairs[reduction_pairs["ri"].isin(shared_rules)].copy()
    coverage_pairs = coverage_pairs[coverage_pairs["ri"].isin(shared_rules)].copy()

    # Use compact integer IDs in the exported coverage file, as in the existing
    # getCoverageLi_pairs output. The expectation identity still joins A to B.
    rule_id_by_identity = {
        identity: rule_id for rule_id, identity in enumerate(sorted(shared_rules))
    }
    for frame in (reduction_rules, coverage_rules, reduction_pairs, coverage_pairs):
        frame["ri"] = frame["ri"].map(rule_id_by_identity)

    output_rows = []
    for precision_index in range(0, 105, 5):
        threshold_started = time.perf_counter()
        precision_required = precision_index / 100
        selected_rules, _, _ = greedy_rule_selection_pairs(
            reduction_pairs,
            reduction_rules,
            precision_required=precision_required,
        )
        selected_set = set(selected_rules)
        b_dirty_pairs = coverage_pairs[
            coverage_pairs["ri"].isin(selected_set) & (coverage_pairs["isDirty"] == 1)
        ]
        rule_to_eval_pairs = {}
        for pair_row in b_dirty_pairs.itertuples(index=False):
            rule_to_eval_pairs.setdefault(pair_row.ri, set())
            for col in str(pair_row.col).split(";"):
                col = col.strip()
                if col:
                    rule_to_eval_pairs[pair_row.ri].add((pair_row.pk, col))

        covered_pairs = set()
        coverage_by_rule = {}
        for rule_id in selected_rules:
            new_pairs = rule_to_eval_pairs.get(rule_id, set()) - covered_pairs
            coverage_by_rule[rule_id] = {"pairs": new_pairs}
            covered_pairs.update(new_pairs)

        b_clean_pairs = coverage_pairs[
            coverage_pairs["ri"].isin(selected_set) & (coverage_pairs["isDirty"] == 0)
        ]
        b_clean_info = coverage_rules[coverage_rules["ri"].isin(selected_set)].loc[:, ["ri", "col_set"]]
        b_clean_merged = pd.merge(b_clean_info, b_clean_pairs, how="inner", on="ri")
        clean_val = len(b_clean_merged.loc[:, ["pk", "col_set"]].drop_duplicates())
        output_rows.append({
            "precValReq": round(precision_index / 10, 2),
            "total_pairs": sum(len(item["pairs"]) for item in coverage_by_rule.values()),
            "clean_val": clean_val,
            "covered_pairs": covered_pairs,
            "selected_rules": selected_rules,
            "coverage_by_rule": coverage_by_rule,
        })
        _paired_log(
            f"Group {group_name}, precision >= {precision_required:.2f}: "
            f"selected {len(selected_rules):,} rules; B covers {len(covered_pairs):,} dirty pairs "
            f"({time.perf_counter() - threshold_started:.1f}s)"
        )

    return output_rows


def run_paired_coverage(data_in, data_out, results_path, pattern_name, reduction_file, coverage_file):
    """Select rules within each original pattern group on A and report coverage on B."""
    _paired_log(
        f"Starting {pattern_name} paired coverage: A={reduction_file}, B={coverage_file}; "
        f"rule results={results_path}; output={data_out}"
    )
    reduction_data = _prepare_paired_dataset(reduction_file, data_in, "reduction")
    coverage_data = _prepare_paired_dataset(coverage_file, data_in, "coverage")
    reduction_groups = _get_paired_pattern_groups(pattern_name, Path(reduction_file).stem)
    coverage_groups = _get_paired_pattern_groups(pattern_name, Path(coverage_file).stem)

    if [group[0] for group in reduction_groups] != [group[0] for group in coverage_groups]:
        raise ValueError("Reduction and coverage pattern groups do not match")

    Path(data_out).mkdir(parents=True, exist_ok=True)
    output_count = 0
    missing_groups = []
    total_groups = len(reduction_groups)
    for group_index, ((group_name, reduction_patterns, reduction_use_glob),
                      (_, coverage_patterns, coverage_use_glob)) in enumerate(
        zip(reduction_groups, coverage_groups), start=1
    ):
        _paired_log(f"Starting pattern group {group_index}/{total_groups}: {group_name}")
        try:
            _, reduction_rules, reduction_pairs, _, _ = _load_paired_rule_results(
                reduction_data, results_path, "reduction",
                reduction_patterns, reduction_use_glob, group_name,
            )
            _, coverage_rules, coverage_pairs, _, _ = _load_paired_rule_results(
                coverage_data, results_path, "coverage",
                coverage_patterns, coverage_use_glob, group_name,
            )
        except FileNotFoundError as exc:
            _paired_log(f"Skipping pattern group {group_name}: {exc}")
            missing_groups.append(group_name)
            continue

        output_rows = _make_paired_coverage_rows(
            reduction_rules, reduction_pairs, coverage_rules, coverage_pairs, group_name
        )
        if output_rows is None:
            _paired_log(f"Skipping pattern group {group_name}: no matching A/B rule definitions")
            missing_groups.append(group_name)
            continue

        output_path = Path(data_out) / (
            f"paired_coverage_{Path(reduction_file).stem}_to_{Path(coverage_file).stem}_"
            f"{pattern_name}_{group_name}.csv"
        )
        pd.DataFrame(output_rows).to_csv(output_path)
        _paired_log(f"Wrote paired coverage results: {output_path}")
        output_count += 1

    if output_count == 0:
        raise FileNotFoundError(
            "No paired result groups were produced. Check PATTERN_NAME, PATTERN_LI/USE_GLOB, "
            "and that both A and B validation results exist for matching rule groups."
        )
    if missing_groups:
        _paired_log(f"Pattern groups without matching A/B results: {missing_groups}")
    _paired_log(f"Finished paired coverage: wrote {output_count}/{total_groups} pattern-group files")


def main():    
    """Run legacy single-dataset evaluation or paired A/B coverage mode."""
    pattern_name = os.environ.get('PATTERN_NAME', 'pci')
    data_in = os.environ.get('DATA_IN', '/data')
    data_out = os.environ.get('DATA_OUT', '/config')
    mypath = os.environ.get('RULE_RESULTS_PATH', '/config')
    
    print(f"Pattern: {pattern_name}")
    print(f"Data In: {data_in}")
    print(f"Data Out: {data_out}")
    print(f"Rule Results Path: {mypath}")

    reduction_file = os.environ.get("REDUCTION_DATA_FILE")
    coverage_file = os.environ.get("COVERAGE_DATA_FILE")
    if reduction_file or coverage_file:
        if not reduction_file or not coverage_file:
            raise ValueError("Set both REDUCTION_DATA_FILE and COVERAGE_DATA_FILE for paired coverage mode")
        run_paired_coverage(data_in, data_out, mypath, pattern_name, reduction_file, coverage_file)
        return
    
    missedPatternDict = dict()
    patternNameLi = ['hospital', 'flights', 'beers', 'tax', 'pci']
    for patternName in patternNameLi:
        if patternName != pattern_name:
            continue
        print(patternName)
        directory = data_in
        match patternName:
            case 'hospital':
                fname = 'hospital_dirty_lab.csv'
            case 'flights':
                fname = 'flights_dirty_lab.csv'
            case 'beers':
                fname = 'beers_dirty_lab.csv'
            case 'tax':
                fname = 'tax_dirty_lab_red.csv'
            case 'pci':
                fname = 'trainGenData_adressPerson_dirtyPol_whole_1000000_25_1.csv'
        
        errInfoDf, maDf, dataDf = getAllInfoDfs(directory, fname)
        
        # Cache row indexes by column so metric calculation can reuse them.
        err_index_by_col = {}
        for row in errInfoDf.itertuples(index=False):
            if row.colName not in err_index_by_col:
                err_index_by_col[row.colName] = set()
            err_index_by_col[row.colName].update(row.index)
        
        ma_index_by_col = {}
        for row in maDf.itertuples(index=False):
            if row.colname not in ma_index_by_col:
                ma_index_by_col[row.colname] = set()
            ma_index_by_col[row.colname].update(row.unexpected_index_list)
        
        colLi = list(dataDf.columns)
        if 'comb' in colLi:
            idx = range(0, colLi.index('comb'))
        elif 'flights' in fname:
            idx = range(0, colLi.index('isDirty'))
        else:
            idx = range(1, colLi.index('isDirty'))
        base_cols = list(dataDf.columns[idx])
        col_names = ['col0'] + base_cols + ['col15', 'col16']
        
        default_patterns = {
            'hospital': ["resDf_all__8_2_26-02-07_18-24-44_new_hospital_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_hospital_dirty_lab_ValiBigOut_",
                        "resDf_all__9_0_26-02-07_18-54-32_new_hospital_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_hospital_dirty_lab_ValiBigOut_",
                        "resDf_all__9_2_26-02-07_19-35-06_new_hospital_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_hospital_dirty_lab_ValiBigOut_",
                        "resDf_all__10_0_26-02-07_20-25-13_new_hospital_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_hospital_dirty_lab_ValiBigOut_",
                        "resDf_all__10_2_26-02-07_20-46-37_new_hospital_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_hospital_dirty_lab_ValiBigOut_"],
            'flights': ["resDf_all__10_0_26-02-02_22-02-10_new_flights_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_flights_dirty_lab_ValiBigOut_",
                       "resDf_all__10_2_26-02-02_22-30-41_new_flights_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_flights_dirty_lab_ValiBigOut_",
                       "resDf_all__8_0_26-02-02_18-04-19_new_flights_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_flights_dirty_lab_ValiBigOut_",
                       "resDf_all__8_2_26-02-02_20-19-23_new_flights_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_flights_dirty_lab_ValiBigOut_",
                       "resDf_all__9_0_26-02-02_20-32-15_new_flights_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_flights_dirty_lab_ValiBigOut_",
                       "resDf_all__9_2_26-02-02_21-25-20_new_flights_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_flights_dirty_lab_ValiBigOut_"],
            'beers': ["resDf_all__10_0_26-02-08_09-12-33_new_beers_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_beers_dirty_lab_ValiBigOut_",
                     "resDf_all__10_2_26-02-08_10-18-17_new_beers_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_beers_dirty_lab_ValiBigOut_",
                     "resDf_all__8_0_26-02-07_21-44-45_new_beers_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_beers_dirty_lab_ValiBigOut_",
                     "resDf_all__8_2_26-02-07_22-45-05_new_beers_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_beers_dirty_lab_ValiBigOut_",
                     "resDf_all__9_0_26-02-08_00-03-18_new_beers_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_beers_dirty_lab_ValiBigOut_",
                     "resDf_all__9_2_26-02-08_04-34-03_new_beers_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_beers_dirty_lab_ValiBigOut_"],
            'tax': ["resDf_all__10_0_26-05-08_21-00-59_new_tax_convertData_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_tax_dirty_lab_red_ValiBigOut_",
                   "resDf_all__8_0_26-05-07_05-11-38_new_tax_convertData_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_tax_dirty_lab_red_ValiBigOut_",
                   "resDf_all__9_0_26-05-07_21-37-10_new_tax_convertData_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_tax_dirty_lab_red_ValiBigOut_",
                   "resDf_all__10_2_26-05-09_02-37-41_new_tax_convertData_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_tax_dirty_lab_red_ValiBigOut_",
                   "resDf_all__8_2_26-05-07_13-41-00_new_tax_convertData_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_tax_dirty_lab_red_ValiBigOut_",
                   "resDf_all__9_2_26-05-08_09-18-44_new_tax_convertData_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_tax_dirty_lab_red_ValiBigOut_"],
            'pci': ["resDf_all__9_0_25-12-31_18-07-13_new_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_trainGenData_adressPerson_dirtyPol_whole_1000000_25_1_ValiBigOut_",
                   "resDf_all__9_2_25-12-28_09-02-29_new_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_trainGenData_adressPerson_dirtyPol_whole_1000000_25_1_ValiBigOut_",
                   "resDf_all__10_0_26-01-01_11-45-53_new_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_trainGenData_adressPerson_dirtyPol_whole_1000000_25_1_ValiBigOut_",
                   "resDf_all__10_2_25-12-29_01-16-08_new_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_trainGenData_adressPerson_dirtyPol_whole_1000000_25_1_ValiBigOut_"]
        }
        
        patternLi = os.environ.get('PATTERN_LI', None)
        if patternLi:
            import ast
            patternLi = ast.literal_eval(patternLi)
        else:
            patternLi = default_patterns.get(patternName, default_patterns['pci'])
        
        useGlob = os.environ.get('USE_GLOB', 'True').lower() == 'true'
        mypath = os.environ.get('RULE_RESULTS_PATH', mypath)
        
        if patternName == 'tax':
            useGlob = True
        patternLi_glob = [pattern for _, pattern in _get_pattern_glob_groups(patternName)]
        if useGlob:
            patternLi = patternLi_glob


        timeLi = []
        missedPatternLi = []
        for pattern in patternLi:
            onlyfiles = getFileLi(mypath, pattern, useGlob)
            if len(onlyfiles) == 0:
                missedPatternLi.append(pattern)
                continue
            if useGlob:
                patternParts = onlyfiles[0].split('/')[-1].split('_')
                idxPat = next(i for i, part in enumerate(patternParts, 1) if part == 'ValiBigOut')

            outFileName = f'resEvalAll_all_NewMetricCalc_test_orjson_{pattern}.csv'

            directory = data_out
            file_exists = os.path.exists(Path(directory)/outFileName)
            successRules_all = 0
            errRules_all = 0
            allRules_all = 0
            isCalcRuleMetric = True
            resLi = list()
            fieldnames = ['ri', 'suiteStr', 'ruleType', 'winStr', 'successRules', 'errRules', 'allRules', 
                        'tp', 'fp', 'fn', 'precision', 'recall', 'laplas_precision', 'laplas_recall', 'col_set', 'fname', 'unexpected_indices']    
            ruleAndPK = []
            ruleInfo = []
            print(f"dirctory: {directory}")
            print(f"pattern: {pattern}")
            dbName = f"groups_all_1_{pattern}_1.db"
            print(f"dbName.split('_'): {dbName.split('_')}")
            print(f"dbName.split('_')[11]: {dbName.split('_')[11]}")
            print(f"dbName.split('_')[12]: {dbName.split('_')[12]}")
            conn, cur, dbName = createDB(directory, pattern)
            print(f"dbName: {dbName}")
            startAll = time.time()
            expectationLi = []
            onlyCoverage = False
            if not onlyCoverage:
                with open(Path(directory)/outFileName, "w", newline="", buffering=1_048_576) as resFile:
                    writer = csv.DictWriter(resFile, fieldnames=fieldnames)
                    writer.writeheader()
                    if not file_exists:
                        writer.writeheader()
                    ri = 0
                    for f in onlyfiles:
                        suiteStr, winStr = getFileInfo(f)
                        if re.search(r'_Size\d+\.txt$', f):
                            lineStr = ''
                            print(f"process {onlyfiles.index(f)} of {len(onlyfiles)-1}: {f}")
                            with open(join(mypath, f), 'r') as file:
                                lineStr = json.load(file)
                                if '[]' != lineStr and '' != lineStr and len(lineStr) > 0:
                                    try:
                                        jsonStr = json.loads(lineStr)
                                    except json.JSONDecodeError:
                                        print(f"  Skipping {f}: contains non-JSON error output")
                                        continue

                                    if isCalcRuleMetric:
                                        successRules, errRules, allRules = countRules(jsonStr)
                                        successRules_all += successRules
                                        errRules_all += errRules
                                        allRules_all += allRules    
                                        for res in jsonStr['results']:
                                            print(f"\t process {onlyfiles.index(f)} of {len(onlyfiles)-1}: {suiteStr} - {winStr} - {ri}")
                                            ruleType = ''
                                            if 'raised_exception' in res['exception_info'].keys() and not res['exception_info']['raised_exception']:
                                                resLi = getMetrics_fast2(res, errInfoDf, maDf, ri, dataDf, 
                                                                         err_index_by_col=err_index_by_col, 
                                                                         ma_index_by_col=ma_index_by_col,
                                                                         isVerbose=False)
                                                tp = resLi[0]
                                                fp = resLi[1]
                                                fn = resLi[2] 
                                                precision = resLi[3]
                                                recall = resLi[4] 
                                                laplas_precision = resLi[5]
                                                laplas_recall = resLi[6]
                                                col_set = resLi[7] 
                                                unexpected_indices = resLi[8]
                                                ruleType = resLi[9]
                                                riAndPKData = resLi[10]
                                                ruleInfo.append((ruleType, ri, ';'.join(col_set), precision, recall))
                                                if len(riAndPKData) > 0:
                                                    ruleAndPK.extend(riAndPKData)
                                                exptStr = res['expectation_config']
                                                typeStr = exptStr['type']
                                                colStr = ', '.join(col_set)
                                                descStr = createDescStr(typeStr, exptStr['kwargs'])
                                                expectationDict = {'type': typeStr, 'column': colStr, 'description': descStr, 'link': f"https://greatexpectations.io/expectations/{typeStr}/"}
                                                expectationLi.append(expectationDict)
                                                resDict = {'ri': ri, 'suiteStr': suiteStr, 'ruleType': ruleType, 'winStr': winStr,
                                                            'successRules': successRules, 'errRules': errRules, 'allRules': allRules, 
                                                            'tp': tp, 'fp': fp, 'fn': fn,
                                                            'precision': precision, 'recall': recall, 
                                                            'laplas_precision': laplas_precision, 'laplas_recall': laplas_recall,
                                                            'col_set': col_set, 
                                                            'fname': f}
                                            else:        
                                                resDict = {'ri': ri, 'suiteStr': suiteStr, 'ruleType': ruleType, 'winStr': winStr,
                                                            'successRules': successRules, 'errRules': errRules, 'allRules': allRules, 
                                                            'tp': None, 'fp': None, 'fn': None,
                                                            'precision': None, 'recall': None, 'laplas_precision': None, 'laplas_recall': None, 'col_set': None, 
                                                            'fname': f}
                                            writer.writerow(resDict)
                                            
                                            ri += 1
                                else:
                                    print(f)
                
                if len(ruleAndPK) > 0:
                    cur.executemany(
                        "INSERT INTO ruleAndPK VALUES (?, ?, ?, ?)",
                        ruleAndPK
                    )
                if len(ruleInfo) > 0:
                    cur.executemany(
                        "INSERT INTO ruleInfo VALUES (?, ?, ?, ?, ?)",
                        ruleInfo
                    )
                conn.commit()
                
                endAll = time.time()
                print(f"res Time:{endAll} - {startAll} = {endAll - startAll} ... {pattern}")
                timeLi.append({'endAll': endAll, 'startAll': startAll, 'diff': endAll - startAll, 'pattern': pattern})
                expectationFileName = f'{Path(outFileName).stem}_expectationLi.csv'
                pd.DataFrame(expectationLi).to_csv(Path(directory)/expectationFileName)


            print('start Coverage Calc')
            precisionValLi = [1, 0.95, 0.9, 0.8, 0.75, 0.5, 0.25, 0.1, 0]
            errorCount = np.zeros((len(dataDf[dataDf['isDirty']].index)+1, len(base_cols)+1))        
            errorCount[:, 0] = np.append(dataDf[dataDf['isDirty']].index, -99)
            indexVec = errorCount[:, 0]
            collArrLi = []
            if 'effortMedium' in dbName:
                promptId = dbName.split('_')[6]
                fileInfo = dbName.split('_')[7]
            else:
                promptId = dbName.split('_')[11]
                fileInfo = dbName.split('_')[12]
            
            for precVal in precisionValLi:
                print(f"Start collArrDf {precVal}")
                if precVal <= 0.1 and pattern == "resDf_all__8_0_25-12-30_10-50-09_new_OnlyRelevanRules_valChange_newValInfo_blub0_BigName_trainGenData_adressPerson_dirtyPol_whole_1000000_25_1_ValiBigOut_":
                    break
                fOutName = f"resEvalAll_coverage_numpy_{promptId}_{fileInfo}_{precVal}.csv"
                try:
                    errorCountRes = getErrCount(conn, cur, precVal, np.copy(errorCount), indexVec, dataDf)
                except Exception as e:
                    print(f"An error occurred: {e}")
                header = ['index'] + list(dataDf.columns[idx])
                np.savetxt(Path(directory)/f'{fOutName}', errorCountRes, delimiter=",", header=','.join(header), comments='')

                arr = np.c_[errorCountRes, np.append(dataDf[dataDf['isDirty']].loc[:, 'errRule'], 'clean'), np.append(dataDf[dataDf['isDirty']].loc[:, 'isDirty'], 'False')]
                collArr = collapseDataframe(arr)
                collArrDf = pd.DataFrame(collArr[:-1, :], columns=col_names)
                collArrDf.loc[:, 'precisionVal'] = precVal
                collArrLi.append(collArrDf)
                
            collArrDfComb = pd.concat(collArrLi, axis=0, ignore_index=True)
            fOutName = f"resEvalAll_collArr_{promptId}_{fileInfo}_{precVal}.csv"
            collArrDfComb.to_csv(Path(directory)/'collArrDf.csv')

            df_pairs = pd.read_sql_query(
                "SELECT ri, pk, col, isDirty FROM ruleAndPK", conn
            )
            df_rules = pd.read_sql_query(
                "SELECT ri, col_set, precision FROM ruleInfo", conn
            )
            coverageLi = getCoverageLi(df_pairs, df_rules)
            pd.DataFrame(coverageLi).to_csv(Path(directory)/f"1_resEvalAll_coverage_Li_{patternName}_{promptId}_{fileInfo}_{precVal}.csv")
            coverageLi = getCoverageLi_pairs(df_pairs, df_rules)
            pd.DataFrame(coverageLi).to_csv(Path(directory)/f"1_resEvalAll_coverage_Li_pairs_{patternName}_{promptId}_{fileInfo}_{precVal}.csv")

            conn.close()

        print(timeLi)
        missedPatternDict[patternName] = missedPatternLi
        print(f"missedPatternDict: {missedPatternDict}")
        for timeRow in timeLi:
            print(f"res Time: {timeRow['endAll']} - {timeRow['startAll']} = {timeRow['diff']} ... {timeRow['pattern']}")
    print(f"missedPatternDict: {missedPatternDict}")


if __name__ == "__main__":
    main()
