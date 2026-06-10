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


def load_latest_per_unique_file(base_dir, prefix='resDf_all__', extension='.csv'):
    """
    Get the latest file for each unique base name.
    
    Args:
        base_dir: Directory to search in
        prefix: File name prefix to match
        extension: File extension to match
        
    Returns:
        Dictionary mapping base names to (full_path, mtime) tuples
    """
    file_groups = defaultdict(list)
    
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith(prefix) and f.endswith(extension):
                full_path = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(full_path)
                    
                    match = re.match(
                        r'resDf_all__\d+_\d+_\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+\.csv)',
                        f
                    )
                    if match:
                        base_name = f"{'_'.join(f.split('_')[3:5])}_{match.group(1)}"
                        file_groups[base_name].append((full_path, mtime))
                except FileNotFoundError:
                    print(f"Warning: File not found: {full_path}")
    
    latest_per_base = {}
    for base_name, files_info in file_groups.items():
        latest = max(files_info, key=lambda x: x[1])
        latest_per_base[base_name] = latest
    
    return latest_per_base


def load_and_process_data(getLatestFiles=True, dirStr=None):
    """
    Load CSV files and process data similar to the R code.
    
    Args:
        getLatestFiles: If True, get the latest file for each unique base name
        dirStr: Directory containing CSV files. If None, uses default paths
        
    Returns:
        Tuple of (sumEvalResAllDf, all_exceptionLi, ruleEvalResAll)
    """
    if dirStr is None:
        dirStr = os.environ.get('EVAL_RESULT_DIR', 'llmTests/res/ruleGen/RuleMetrik/server/all/')
        
        alt_paths = [
            'llmTests/res/ruleGen/RuleMetrik/',
            'data/rule_eval/output/',
            'C:\\Users\\glock\\Documents\\Projekt\\DANTE\\QuanTD\\code\\llmTests\\res\\ruleGen\\RuleMetrik\\'
        ]
        
        if not os.path.exists(dirStr):
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    dirStr = alt_path
                    break
    
    if getLatestFiles:
        fnameVec = load_latest_per_unique_file(dirStr)
        fnameVec = [fnameVec[x][0] for x in fnameVec if 'ExcecPara' not in x and 'resTimeLi' not in x]
    else:
        fnameVec = [f for f in os.listdir(dirStr) if f.endswith('.csv') and f.startswith('resDf_')]
    
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
        
        parts = fname_display.split('_')
        promptId = parts[3] if len(parts) > 3 else 'unknown'
        ruleInfo = parts[4] if len(parts) > 4 else 'unknown'
        
        if len(parts) == 12:
            datasetName = 'PCI'
        else:
            datasetName = parts[8] if len(parts) > 8 else 'unknown'
        
        promptInfo_map = {
            '0': 'Dirty & Clean',
            '1': 'Just Dirty',
            '2': 'Dirty & Class'
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
    
    grouped = sumEvalResAllDf_filtered.groupby(['fileInfo_clean', 'Modelname_short', 'datasetName', 'errRule'])
    
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
        categories=['Just Dirty', 'Dirty & Class', 'Dirty & Clean'],
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


def create_model_comparison_plot(sumEvalResAllDf_filtered, dataset_name='PCI', output_dir=None):
    """
    Create a model comparison scatter plot (Precision vs Recall per rule) for a specific dataset.
    
    Args:
        sumEvalResAllDf_filtered: Transformed evaluation results dataframe
        dataset_name: Name of the dataset to filter (default: 'PCI')
        output_dir: Directory to save plots. If None, uses default path
        
    Returns:
        plotnine ggplot object
    """
    from plotnine import ggplot, geom_point, aes, facet_grid, theme_bw, labs, theme, element_text, scale_color_discrete, scale_shape_discrete, ylab, xlab
    
    # Filter for the specific dataset
    df = sumEvalResAllDf_filtered[sumEvalResAllDf_filtered['datasetName'] == dataset_name].copy()
    
    # Map prompt IDs to descriptions
    prompt_map = {
        '0': 'Dirty & Clean',
        '1': 'Dirty & Class',
        '2': 'Just Dirty'
    }
    df['promptId'] = df['promptId'].astype(str).replace(prompt_map)
    
    # Create the plot
    p = (
        ggplot(df, aes(x='xVal', y='value', color='variable', shape='variable'))
        + geom_point(size=2)
        + facet_grid('promptId ~ Modelname_short', scales='free')
        + theme_bw()
        + ylab('Precision and Recall')
        + xlab('Rule')
        + labs(color='Metric', shape='Metric')
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
        + scale_color_discrete(labels={
            'recall_mean': 'Recall',
            'precision_mean': 'Precision'
        })
        + scale_shape_discrete(labels={
            'recall_mean': 'Recall',
            'precision_mean': 'Precision'
        })
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
            '1': 'Dirty & Class',
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
        temp_df['promptId'] = temp_df['promptId'].replace('Just Dirty').replace('Dirty & Class').replace('Dirty & Clean')
        temp_df['promptId'] = temp_df['promptId'].astype(str).replace({
            '0': 'Dirty & Clean',
            '1': 'Just Dirty', 
            '2': 'Dirty & Class'
        })
        temp_df['ruleInfoText'] = temp_df['ruleInfoText'].replace('noParam', 'No Parameter').replace('Param', 'Parameter')
        
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
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory containing evaluation result CSV files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--dataset', type=str, default='PCI',
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
    create_model_comparison_plot(
        sumEvalResAllDf_filtered,
        dataset_name=args.dataset,
        output_dir=args.output_dir
    )