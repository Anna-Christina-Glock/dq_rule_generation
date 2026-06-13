"""
Rule Generation Main Script (Parallel Processing)

This script runs the LLM-based rule generation process in parallel using threads.
"""

import os
import numpy as np
import pandas as pd
import re
import json
from pathlib import Path
from datetime import datetime
import threading
import logging

from llm_utils import LLMClient
from rule_gen_prompt import generate_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ask_model_save_answer(
    llm_package: str,
    client: LLMClient,
    modelname: str,
    prompt_id: int,
    clean_df: pd.DataFrame,
    polluted_df: pd.DataFrame,
    res_list: list,
    index: int,
    expectation_list_id: int = 0,
    n_cols: int = 14,
    id_column: str = "pk",
    error_detection_info: str = None,
):
    """
    Process a single row: generate prompt, ask model, and save result.
    
    Args:
        llm_package: LLM package name ("ollama" or "openai")
        client: LLMClient instance
        modelname: Model name to use
        prompt_id: Prompt template ID
        clean_df: Clean DataFrame
        polluted_df: Polluted DataFrame
        res_list: List to append results to (thread-safe)
        index: Row index to process
        expectation_list_id: ID for expectation list
        n_cols: Number of columns to include
        id_column: ID column name
        error_detection_info: Additional error detection info
    """
    try:
        polluted_row = polluted_df.iloc[[index]]
        prompt_str = generate_prompt(
            clean_df, 
            polluted_row,
            n_rows=1, 
            prompt_id=prompt_id,
            expectation_list_id=expectation_list_id,
            n_cols=n_cols,
            id_column=id_column,
            error_detection_info=error_detection_info
        )
        
        # Strip whitespace from lines
        lines = prompt_str.split('\n')
        stripped_lines = [line.strip() for line in lines]
        prompt_str = '\n'.join(stripped_lines)
        
        logger.info(f"[{index}] Before asking model")
        response = client.ask_model(prompt_str, modelname)
        logger.info(f"[{index}] After asking model")
        
        # Create result dictionary
        res_dict = polluted_row.iloc[0].to_dict()
        res_dict.update({
            'prompt': prompt_str,
            'answer': response,
            'modelname': modelname,
            'prompt_id': prompt_id,
            'expectation_list_id': expectation_list_id,
            'row_index': index
        })
        res_list.append(res_dict)
        
    except Exception as e:
        logger.error(f"Error processing row {index}: {e}")
        raise


def main_rule_gen(
    config_path: Path,
    connection_config_path: Path = None,
    model_config_name: str = None,
    override_params: dict = None,
):
    """
    Main function to run the rule generation pipeline.
    
    Args:
        config_path: Path to parameter configuration JSON
        connection_config_path: Path to LLM connection configuration JSON
        model_config_name: Name of the model config to use (from connection config)
        override_params: Dictionary to override config values
    """
    # Load parameter configuration
    with open(config_path) as param_file:
        parameter_config = json.load(param_file)
    
    # Apply overrides if provided
    if override_params:
        parameter_config.update(override_params)
    
    # Determine model config name
    if model_config_name is None:
        model_config_name = parameter_config.get("modelConfigName", "default")
    
    # Load LLM connection configuration
    if connection_config_path is None:
        # Try default locations
        for path in ["configs/connection.json", "connection.json", "../configs/connection.json"]:
            conn_path = Path(path)
            if conn_path.exists():
                connection_config_path = conn_path
                break
    
    if connection_config_path is None:
        raise FileNotFoundError("Could not find connection configuration file")
    
    with open(connection_config_path) as config_file:
        llm_connection_config = json.load(config_file)
    
    # Get model configuration
    if model_config_name not in llm_connection_config:
        raise ValueError(f"Model config '{model_config_name}' not found in connection config")
    
    llm_package = llm_connection_config[model_config_name]["llm-package"]
    api_key = llm_connection_config[model_config_name].get("api_key")
    base_url = llm_connection_config[model_config_name].get("base_url")
    modelname = llm_connection_config[model_config_name]["modelname"]
    
    # Initialize LLM client
    logger.info("Connecting to LLM client...")
    client = LLMClient(llm_package, api_key=api_key, base_url=base_url)
    logger.info(f"Connected to model: {modelname}")
    
    # Get location parameter
    param_init_key = parameter_config["location"]
    file_idx_arr = parameter_config[param_init_key]["File"]["Input"]
    
    for file_idx in file_idx_arr:
        input_file_param = parameter_config["Files"][file_idx]
        data_folder = Path(parameter_config[param_init_key]["Paths"]["Data"])
        
        # Read data files
        csv_df_dirty = pd.read_csv(
            data_folder / input_file_param['name'],
            encoding=None if input_file_param.get('encoding', 'None') == 'None' else input_file_param['encoding'],
            delimiter=None if input_file_param.get('delimiter', 'None') == 'None' else input_file_param['delimiter']
        )
        
        csv_df_clean = pd.read_csv(
            data_folder / input_file_param['cleanFile'],
            encoding=None if input_file_param.get('encoding', 'None') == 'None' else input_file_param['encoding'],
            delimiter=None if input_file_param.get('delimiter', 'None') == 'None' else input_file_param['delimiter'],
            dtype='object'
        )
        csv_df_clean['pk'] = csv_df_clean['pk'].apply(lambda x: int(x))
        
        # Get parameters
        n_cols = input_file_param.get('n_cols', 14)
        id_column = input_file_param.get('id_column', 'pk')
        error_detection_info = input_file_param.get('err_dect_info')
        
        logger.info(f"Processing file: {input_file_param['name']}")
        
        result_folder = Path(parameter_config[param_init_key]["Paths"]["Result"])
        res_file_suffix = parameter_config["Files"][file_idx].get("OutputSuffix", "")
        
        # Filter for dirty rows
        df_fil = csv_df_dirty.query("isDirty == True")
        
        # Prompt IDs to run
        prompt_id_list = parameter_config.get("promptIds", [0, 1, 2])
        
        for prompt_id_val in prompt_id_list:
            res_list = list()
            is_header = True
            mode_val = 'w'
            
            # Expectation list IDs
            expectation_list_ids = parameter_config.get("expectationListIds", [0, 2])
            
            for expectation_list_id in expectation_list_ids:
                offset_str = f"_{prompt_id_val}_{expectation_list_id}_{datetime.now().strftime('%y-%m-%d_%H-%M-%S')}"
                polluted_df = df_fil
                
                # Thread configuration
                thread_count = parameter_config.get("threadCount", 52)
                if thread_count > len(csv_df_dirty):
                    thread_count = len(csv_df_dirty)
                threads = [None] * thread_count
                i = 0
                
                if len(csv_df_clean) <= len(csv_df_dirty):
                    logger.info(f"Processing {len(csv_df_clean)} rows with batch size {thread_count}")
                    processDf = csv_df_clean
                else:
                    logger.info(f"Processing {len(csv_df_dirty)} rows with batch size {thread_count}")
                    processDf = csv_df_dirty

                for index, row in processDf.iterrows():
                    logger.info(f'Row: {index}')
                    
                    threads[i] = threading.Thread(
                        target=ask_model_save_answer,
                        args=(
                            llm_package, client, modelname, prompt_id_val,
                            csv_df_clean, polluted_df, res_list, index,
                            expectation_list_id, n_cols, id_column, error_detection_info
                        )
                    )
                    threads[i].start()
                    i += 1
                    
                    # Wait for batch to complete
                    if (index % (thread_count - 1) == 0) and index > 1:
                        logger.info(f"Batch complete at index {index}")
                        for j in range(len(threads)):
                            if threads[j] is None:
                                break
                            threads[j].join()
                        
                        # Save batch results
                        df_all = pd.DataFrame(res_list)
                        df_all = df_all.assign(modelname=modelname)
                        df_all.to_csv(
                            result_folder / f'resDf_all_{offset_str}_{res_file_suffix}.csv',
                            mode=mode_val,
                            header=is_header,
                            index=False
                        )
                        res_list = []
                        is_header = False
                        mode_val = 'a'
                        threads = [None] * len(threads)
                        i = 0
                
                # Join remaining threads
                for j in range(len(threads)):
                    if threads[j] is None or i == 0:
                        break
                    threads[j].join()
                
                # Save final results
                df_all = pd.DataFrame(res_list)
                df_all = df_all.assign(modelname=modelname)
                df_all.to_csv(
                    result_folder / f'resDf_all_{offset_str}_{res_file_suffix}.csv',
                    mode=mode_val,
                    header=is_header,
                    index=False
                )
        
        logger.info(f"-------------- Finished processing file: {input_file_param['name']} --------------")
    
    logger.info('-------------- All files processed --------------')


if __name__ == "__main__":
    main_rule_gen(
        config_path=Path("configs/parameter.json"),
        connection_config_path=Path("configs/connection.json")
    )