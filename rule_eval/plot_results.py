"""
Rule Evaluation Result Visualization

This module provides functions for loading, processing, and visualizing
LLM-generated rule evaluation results.
"""

import os
import re
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from collections import defaultdict


def getF1(precision, recall):
    """
    Calculate F1 score, returning 0 when precision+recall=0.
    
    Args:
        precision: Precision values (scalar or array)
        recall: Recall values (scalar or array)
        
    Returns:
        F1 score values (same shape as inputs)
    """
    f1 = np.where((precision + recall) == 0, 0,
                  2 * precision * recall / (precision + recall))
    return f1

def load_and_process_data(getLatestFiles=True, dirStr=None):
    """
    Load CSV files and process data similar to the R code.
    
    Args:
        getLatestFiles: If True, get the latest file for each unique base name
        dirStr: Directory containing CSV files. If None, uses default paths
        
    Returns:
        Tuple of (sumEvalResAllDf, all_exceptionLi, ruleEvalResAll)
    """    
    if getLatestFiles:
        fnameVec = load_latest_per_unique_file(dirStr)
        fnameVec = [fnameVec[x][0] for x in fnameVec if 'ExcecPara' not in x and 'resTimeLi' not in x]
    else:
        fnameVec = [f for f in os.listdir(dirStr) if f.endswith('.csv') and f.startswith('resDf_') and 'ExcecPara' not in f and 'resTimeLi' not in f]
    
    print(f"dirStr: {dirStr}")
    print(f"Found {len(fnameVec)} CSV files")
    
    sumEvalResAllDf = pd.DataFrame()
    all_exceptionLi = []
    ruleEvalResAll = pd.DataFrame()
    
    for fname_idx, fname in enumerate(fnameVec):
        print(f"Processing: {fname}")
        
        if getLatestFiles:
            fPath = fname
            ruleEvalRes = pd.read_csv(fPath)
            fname_display = os.path.basename(fname)
        else:
            ruleEvalRes = pd.read_csv(os.path.join(dirStr, fname))
            fname_display = fname
        
        # Key mapping for camelCase (legacy) to snake_case (new)
        # This allows the plotter to work with both old and new file formats
        key_mapping = {
            'rule_Count': 'rule_count',
            'precision_perRow': 'precision_per_row',
            'recall_perRow': 'recall_per_row',
            'precision_perRuleCol': 'precision_per_rule_col',
            'recall_perRuleCol': 'recall_per_rule_col',
            'errRule': 'err_rule',
            'numErr': 'num_err',
            'numPerRule': 'num_per_rule'
        }
        
        # Rename columns if snake_case versions exist
        for old_key, new_key in key_mapping.items():
            if old_key not in ruleEvalRes.columns and new_key in ruleEvalRes.columns:
                ruleEvalRes = ruleEvalRes.rename(columns={new_key: old_key})
        
        # Convert key metric columns to numeric to prevent string arithmetic errors
        metric_cols = ['precision_perRow', 'recall_perRow', 'precision_perRuleCol', 'recall_perRuleCol']
        for col in metric_cols:
            if col in ruleEvalRes.columns:
                ruleEvalRes[col] = pd.to_numeric(ruleEvalRes[col], errors='coerce')
        
        parts = fname_display.split('_')
        promptId = parts[3] if len(parts) > 3 else 'unknown'
        ruleInfo = parts[4] if len(parts) > 4 else 'unknown'
        
        if len(parts) == 12:
            datasetName = 'PCI'
        else:
            datasetName = parts[8] if len(parts) > 8 else 'unknown'
        
        promptInfo_map = {
            '8': 'Dirty & Clean',
            '9': 'Just Dirty',
            '10': 'Dirty & Type'
        }
        promptInfo = promptInfo_map.get(promptId, f'unknown_{promptId}')
        
        ruleInfo_map = {
            '0': 'noParam',
            '2': 'Param'
        }
        ruleInfoText = ruleInfo_map.get(ruleInfo, f'unknown_{ruleInfo}')
        
        rule_Count_sum = ruleEvalRes['rule_Count'].sum()
        exception_count_sum = ruleEvalRes['exception_count'].sum()
        
        ruleEvalRes['numErr'] = ruleEvalRes['errRule'].apply(
            lambda x: len(str(x).split(';')) - 1 if pd.notna(x) else 0
        )
        
        ruleEvalRes['f1_perRow'] = getF1(ruleEvalRes['precision_perRow'], ruleEvalRes['recall_perRow'])
        ruleEvalRes['f1_perRuleCol'] = getF1(ruleEvalRes['precision_perRuleCol'], ruleEvalRes['recall_perRuleCol'])
        
        # Add numPerRule (count per errRule) - this is calculated, not in the raw data
        ruleEvalRes['numPerRule'] = ruleEvalRes.groupby('errRule')['errRule'].transform('count')
        
        grouped = ruleEvalRes.groupby(['errRule', 'numErr', 'numPerRule'])
        
        sumEvalResDf = grouped.agg({
            'f1_perRuleCol': ['mean', 'std'],
            'f1_perRow': ['mean', 'std'],
            'precision_perRuleCol': ['mean', 'std'],
            'recall_perRuleCol': ['mean', 'std'],
            'precision_perRow': ['mean', 'std'],
            'recall_perRow': ['mean', 'std'],
        }).reset_index()
        
        sumEvalResDf.columns = ['errRule', 'numErr', 'numPerRule',
                                'mean_f1_per_errRule', 'sd_f1_per_errRule',
                                'mean_f1_per_errRow', 'sd_f1_per_errRow',
                                'mean_precision_per_errRule', 'sd_precision_per_errRule',
                                'mean_recall_per_errRule', 'sd_recall_per_errRule',
                                'mean_precision_per_errRow', 'sd_precision_per_errRow',
                                'mean_recall_per_errRow', 'sd_recall_per_errRow']
        
        sumEvalResDf['fileInfo'] = f'{promptInfo}_{ruleInfoText}'
        
        temp_df = sumEvalResDf[['numErr', 'errRule',
                                'mean_f1_per_errRule', 'mean_f1_per_errRow',
                                'mean_precision_per_errRule', 'mean_recall_per_errRule',
                                'mean_precision_per_errRow', 'mean_recall_per_errRow',
                                'fileInfo']].copy()
        temp_df['fname'] = fname_display
        temp_df['datasetName'] = datasetName
        temp_df['prompt'] = promptInfo
        temp_df['fid'] = fname_idx + 1
        
        modelname_long = ruleEvalRes['modelname'].iloc[0] if len(ruleEvalRes) > 0 else 'unknown'
        modelname_short = ''
        
        if 'cyankiwi/GLM-4.7-Flash-AWQ-4bit' in modelname_long:
            modelname_short = 'GLM-4.7'
        elif 'cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit' in modelname_long:
            modelname_short = 'Qwen3-Coder'
        elif 'RedHatAI/gemma-4-31B-it-FP8-Dynamic' in modelname_long:
            modelname_short = 'Gemma-4'
        
        temp_df['Modelname_long'] = modelname_long
        temp_df['Modelname_short'] = modelname_short
        
        exceptionDict = {
            'Modelname_short': modelname_short,
            'datasetName': datasetName,
            'ruleInfo': ruleInfo,
            'ruleInfoText': ruleInfoText,
            'promptId': promptId,
            'promptInfo': promptInfo,
            'rule_Count_sum': rule_Count_sum,
            'exception_count_sum': exception_count_sum
        }
        all_exceptionLi.append(exceptionDict)
        
        sumEvalResAllDf = pd.concat([sumEvalResAllDf, temp_df], ignore_index=True)
        ruleEvalRes['Modelname_short'] = modelname_short
        ruleEvalRes['fname'] = fname_display
        ruleEvalRes['datasetName'] = datasetName
        ruleEvalRes['prompt'] = promptInfo
        ruleEvalRes['fid'] = fname_idx + 1
        ruleEvalResAll = pd.concat([ruleEvalResAll, ruleEvalRes], ignore_index=True)
    
    return sumEvalResAllDf, all_exceptionLi, ruleEvalResAll


def transform_data_for_plotting(sumEvalResAllDf):
    """
    Transform data for plotting similar to R's dplyr + tidyr operations.
    
    Args:
        sumEvalResAllDf: Raw evaluation results dataframe
        
    Returns:
        Transformed dataframe ready for plotting
    """
    sumEvalResAllDf_filtered = sumEvalResAllDf[sumEvalResAllDf['numErr'] == 1].copy()
    
    sumEvalResAllDf_filtered['fileInfo_clean'] = sumEvalResAllDf_filtered['fileInfo'].apply(
        lambda x: x.split('_')[0] if len(x.split('_')) > 0 else x
    )
    sumEvalResAllDf_filtered['fileInfo_clean'] = sumEvalResAllDf_filtered['fileInfo_clean'].replace({
        '&': ' & ',
        'Just Dirty': 'Just Dirty',
        'justDirty': 'Just Dirty'
    })
    
    # Ensure promptId column exists for plotting
    if 'promptId' not in sumEvalResAllDf_filtered.columns and 'prompt' in sumEvalResAllDf_filtered.columns:
        # Map prompt descriptions back to IDs for plotting
        prompt_id_map = {'Dirty & Clean': '0', 'Dirty & Type': '1', 'Just Dirty': '2'}
        sumEvalResAllDf_filtered['promptId'] = sumEvalResAllDf_filtered['prompt'].map(prompt_id_map)
    
    grouped = sumEvalResAllDf_filtered.groupby(['fileInfo_clean', 'Modelname_short', 'datasetName', 'errRule', 'promptId'])
    
    sumEvalResAllDf_filtered_agg = grouped.agg({
        'mean_f1_per_errRule': 'mean',
        'mean_f1_per_errRow': 'mean',
        'mean_precision_per_errRule': 'mean',
        'mean_recall_per_errRule': 'mean',
        'mean_precision_per_errRow': 'mean',
        'mean_recall_per_errRow': 'mean',
    }).reset_index()
    
    sumEvalResAllDf_filtered_agg['fileInfo_cat'] = pd.Categorical(
        sumEvalResAllDf_filtered_agg['fileInfo_clean'],
        categories=['Just Dirty', 'Dirty & Type', 'Dirty & Clean'],
        ordered=True
    )
    
    errRule_map = {
        'DomainViolation': 'Domain Violation',
        'IncorrectEncoding': 'Incorrect Encoding',
        'IncorrectFormat': 'Incorrect Format',
        'MisfieldedValue': 'Misfielded Value',
        'Contradictions': 'Contradictions',
        'DisguisedMissingValues': 'Disguised Missing Values',
        'EmbeddedValue': 'Embedded Value',
        'ExplicitMissingValue': 'Explicit Missing Value',
        'SpellingMistake': 'Spelling Mistake'
    }
    sumEvalResAllDf_filtered_agg['errRule_clean'] = sumEvalResAllDf_filtered_agg['errRule'].replace(errRule_map)
    
    errRule_str_map = {
        ';explicitMissingValue': 'emv',
        ';disguisedMissingValues': 'dmv',
        ';contradictions': 'con',
        ';Misfielded_Value': 'mfv',
        ';embeddedValue': 'ebv',
        ';spellingMistake': 'spm',
        ';Domain_Violation': 'dov',
        ';Incorrect_Format': 'ifo',
        ';Incorrect_Encoding': 'ics',
        'Explicit Missing Value': 'emv',
        'Disguised Missing Values': 'dmv',
        'Contradictions': 'con',
        'Misfielded Value': 'mfv',
        'Embedded Value': 'ebv',
        'Spelling Mistake': 'spm',
        'Domain Violation': 'dov',
        'Incorrect Format': 'ifo',
        'Incorrect Encoding': 'ics'
    }
    sumEvalResAllDf_filtered_agg['errRuleStr'] = sumEvalResAllDf_filtered_agg['errRule_clean'].replace(errRule_str_map)
    
    return sumEvalResAllDf_filtered_agg


def create_model_comparison_plot(sumEvalResAllDf_filtered, dataset_name='PCI', output_dir=None, use_f1=True):
    """
    Create a model comparison scatter plot for a specific dataset.
    Matches logic from rule_eval_all_plots_clean.ipynb.
    
    Args:
        sumEvalResAllDf_filtered: Transformed evaluation results dataframe
        dataset_name: Name of the dataset to filter (default: 'PCI')
        output_dir: Directory to save plots. If None, uses default path
        use_f1: If True, plots F1 per Row. If False, plots Precision/Recall per Row.
        
    Returns:
        plotnine ggplot object
    """
    from plotnine import ggplot, geom_point, aes, facet_grid, theme_bw, labs, theme, element_text, scale_color_manual, scale_shape_manual, ylab, xlab
    
    # Filter for the specific dataset (case-insensitive)
    df = sumEvalResAllDf_filtered[sumEvalResAllDf_filtered['datasetName'].str.lower() == dataset_name.lower()].copy()
    
    # Check if data is empty and provide helpful error
    if len(df) == 0:
        print(f"Error: No data found for dataset '{dataset_name}'. Available datasets in results: {sumEvalResAllDf_filtered['datasetName'].unique().tolist()}")
        return None
    
    
    # Map prompt IDs to descriptions
    prompt_map = {
        '0': 'Dirty & Clean',
        '1': 'Dirty & Type',
        '2': 'Just Dirty',
        '8': 'Dirty & Clean',
        '9': 'Dirty & Type',
        '10': 'Just Dirty'
    }
    df['promptId'] = df['promptId'].astype(str).replace(prompt_map)
    
    # Determine which metrics to plot based on use_f1
    if use_f1:
        value_vars = ['mean_f1_per_errRow']
        y_label = 'F1'
    else:
        value_vars = ['mean_precision_per_errRow', 'mean_recall_per_errRow']
        y_label = 'Precision / Recall'
    
    # Melt the dataframe to long format for plotnine
    # We use errRuleStr as the X axis to match the notebook
    id_vars = ['errRuleStr', 'promptId', 'Modelname_short', 'errRule']
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='variable',
        value_name='value'
    )
    
    # Define model-based colors and shapes to match notebook
    model_colors = {'GLM-4.7': '#1f77b4', 'Qwen3-Coder': '#ff7f0e', 'Gemma-4': '#2ca02c'}
    model_shapes = {'GLM-4.7': 'o', 'Qwen3-Coder': '^', 'Gemma-4': 'v'}
    
    # Create the plot
    p = (
        ggplot(df_long, aes(x='errRuleStr', y='value', color='Modelname_short', shape='Modelname_short'))
        + geom_point(size=3, stroke=2)
        + facet_grid('promptId ~ Modelname_short', scales='free')
        + theme_bw()
        + ylab(y_label)
        + xlab('Error Class')
        + labs(color='Model', shape='Model')
        + theme(
            axis_title_x=element_text(size=14),
            axis_title_y=element_text(size=14),
            axis_text_x=element_text(angle=45, hjust=1, vjust=1, size=11),
            axis_text_y=element_text(size=11),
            strip_text=element_text(size=11),
            legend_title=element_text(size=12),
            legend_text=element_text(size=12),
            figure_size=(12, 5.8),
            legend_position='bottom',
        )
        + scale_color_manual(values=model_colors)
        + scale_shape_manual(values=model_shapes)
    )
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        p.save(os.path.join(output_dir, f'model_comparison_{dataset_name.lower()}.jpg'), width=12, height=5.8, dpi=100)
        print(f"Saved: {output_dir}model_comparison_{dataset_name.lower()}.jpg")
    
    return p


def create_coverage_plot_models(dfAll_other_perc, output_dir=None):
    """
    Create coverage plot showing tradeoff between minimum precision and coverage (dirty vs clean rows).
    
    Args:
        dfAll_other_perc: DataFrame with total_coverage_agg_perc and clean_val_agg_perc columns
        output_dir: Directory to save plots. If None, uses default path
        
    Returns:
        Tuple of (ggplot object for display, ggplot object with custom labels)
    """
    from plotnine import ggplot, geom_line, aes, facet_grid, theme_bw, labs, theme, element_text, scale_linetype_discrete, scale_color_discrete, ylab, xlab, guide_legend
    
    # Melt to long format
    df_long = dfAll_other_perc.melt(
        id_vars=['promptId', 'precValReq', 'dataset', 'Modelname_short'],
        value_vars=['total_coverage_agg_perc', 'clean_val_agg_perc'],
        var_name='variable',
        value_name='value'
    )
    df_long = df_long.astype({'promptId': str})
    
    # Create base plot
    p9 = (
        ggplot(df_long, aes(x='precValReq', y='value', linetype='promptId', color='variable'))
        + geom_line()
        + facet_grid(cols='dataset', rows='Modelname_short')
        + theme_bw()
        + ylab('Percent of Covered Samples (Dirty + Clean)')
        + xlab('Minimum Precision Score of Selected Rules')
        + guides(
            color=guide_legend(ncol=1),
            linetype=guide_legend(ncol=1)
        )
        + theme( 
            legend_position='bottom'
        )
    )
    
    # Apply legend labels and styling
    p9_leg = (
        p9
        + scale_linetype_discrete(labels={
            '0': 'Dirty & Clean',
            '1': 'Dirty & Type',
            '2': 'just Dirty'
        })
        + scale_color_discrete(labels={
            'clean_val_agg_perc': 'Clean Rows',
            'total_coverage_agg_perc': 'Dirty Rows'
        })
        + guides(
            linetype=guide_legend(title='Error Information'),
            color=guide_legend(title='Covered Rows')
        )
        + theme(
            axis_title_x=element_text(size=14),
            axis_title_y=element_text(size=14),
            axis_text_x=element_text(size=11),
            axis_text_y=element_text(size=11),
            strip_text=element_text(size=13),
            legend_title=element_text(size=12),
            legend_text=element_text(size=12),
            figure_size=(15, 7),
        )
    )
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        p9_leg.save(os.path.join(output_dir, 'coverage_plot_models.jpg'), width=15, height=7, dpi=100)
        print(f"Saved: {output_dir}coverage_plot_models.jpg")
    
    return p9, p9_leg


def export_execution_stats_csv(all_exceptionLi, output_dir=None):
    """
    Export rule execution statistics to a CSV format compatible with R's rule_execution_status.R.
    
    Args:
        all_exceptionLi: List of exception dictionaries from load_and_process_data()
        output_dir: Directory to save CSV. If None, saves to 'data/rule_eval/output/'
        
    Returns:
        Path to the saved CSV file
    """
    # Process exception data into a format similar to R's all_exceptionLi_1_all
    allLi = []
    
    for i, exceptionDict in enumerate(all_exceptionLi):
        temp_df = pd.DataFrame([exceptionDict])
        temp_df['All Rules'] = temp_df['rule_Count_sum']
        temp_df['Not Executable Rules'] = temp_df['exception_count_sum']
        temp_df['Executable Rules'] = temp_df['rule_Count_sum'] - temp_df['exception_count_sum']
        temp_df['Not Executable Rules P'] = (temp_df['exception_count_sum'] / temp_df['rule_Count_sum']) * 100
        temp_df['Executable Rules P'] = (temp_df['Executable Rules'] / temp_df['All Rules']) * 100
        
        # Map promptId and ruleInfoText
        temp_df['promptId'] = temp_df['promptId'].astype(str).replace({
            'Dirty & Clean': '0', 'Dirty & Type': '1', 'Just Dirty': '2'
        })
        temp_df['ruleInfoText'] = temp_df['ruleInfoText'].replace({
            'noParam': 'No Parameter', 'Param': 'Parameter'
        })
        
        # Add datasetName (convert to proper case)
        temp_df['datasetName'] = [f"{x[0].upper()}{x[1:]}" for x in temp_df['datasetName']]
        
        allLi.append(temp_df)
    
    result_df = pd.concat(allLi, ignore_index=True)
    
    # Add Count_round for R script compatibility
    result_df['Count_round'] = round(result_df['All Rules'], 1)
    
    # Melt for plotting
    plot_df = result_df.melt(
        id_vars=['datasetName', 'promptId', 'ruleInfoText', 'Modelname_short'],
        value_vars=['Executable Rules', 'Not Executable Rules', 'Executable Rules P', 'Not Executable Rules P'],
        var_name='Rule_Type',
        value_name='Count'
    )
    
    # Map Rule_Type values
    plot_df['Rule_Type'] = plot_df['Rule_Type'].replace({
        'Executable Rules': 'Executable',
        'Not Executable Rules': 'Not Executable',
        'Executable Rules P': 'Executable %',
        'Not Executable Rules P': 'Not Executable %'
    })
    
    if output_dir is None:
        output_dir = 'data/rule_eval/output/'
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'step1_plot_df.csv')
    plot_df.to_csv(output_path, index=False)
    
    print(f"Saved execution statistics to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate rule evaluation plots')
    parser.add_argument('--input-dir', type=str, default="pub/dq_rule_generation/data/results_eval",
                        help='Directory containing evaluation result CSV files')
    parser.add_argument('--output-dir', type=str, default="plots",
                        help='Directory to save plots')
    parser.add_argument('--dataset', type=str, default='hospital',
                        help='Dataset name for model comparison plot')
    parser.add_argument('--latest-only', action='store_true',
                        help='Only process the latest file for each dataset')
    
    args = parser.parse_args()
    
    # Load and process data
    sumEvalResAllDf, all_exceptionLi, ruleEvalResAll = load_and_process_data(
        getLatestFiles=args.latest_only,
        dirStr=args.input_dir
    )
    
    # Transform data for plotting
    sumEvalResAllDf_filtered = transform_data_for_plotting(sumEvalResAllDf)
    
    # Add xVal column for plotting
    sumEvalResAllDf_filtered['xVal'] = range(0, len(sumEvalResAllDf_filtered))
    
    # Export execution stats for R script
    export_execution_stats_csv(all_exceptionLi, output_dir=args.output_dir)
    
    # Create and save plots
    plot_result = create_model_comparison_plot(
        sumEvalResAllDf_filtered,
        dataset_name=args.dataset,
        output_dir=args.output_dir
    )
    
    if plot_result is None:
        print("Plot creation failed. Check the error message above.")
