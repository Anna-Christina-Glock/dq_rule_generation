import json
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from adaptiveConcurrency import AdaptiveController, AdaptiveGate
from llmUtils import askModelWithMetrics, getClient
from ruleGenPrompt import generate_prompt

def askModelSaveAnswer(llm_package,client,modelname,promptIdVal,
                       cleanDf,pollutedDf,file_content,resLi,i,expectationLiId=0,
                       n_cols=14,id_column='pk',err_dect_info=None,
                       reasoning_config=None,modelConfigName=None,reasoningTag=None,
                       gate=None,controller=None):
        promptStr = generate_prompt(cleanDf,pollutedDf.iloc[[i]],n_rows=1,promptId=promptIdVal,fileContent=file_content,expectationLiId=expectationLiId,
                                    n_cols=n_cols,id_column=id_column,err_dect_info=err_dect_info)
        lines = promptStr[0].split('\n')
        stripped_lines = [line.strip() for line in lines]
        #print('\n'.join(stripped_lines))
        promptStr = '\n'.join(stripped_lines)
        #print(modelname)
        print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - {i}: before Ask")
        in_flight = gate.acquire() if gate is not None else None
        res = None
        errStr = ""
        metrics = {"ttft_s": None, "total_time_s": None, "tokens": None, "tokens_per_s": None}
        try:
            res, metrics = askModelWithMetrics(llm_package,client,promptStr,modelname,reasoning_config=reasoning_config)
        except Exception as e:
            # the request failed - keep going with the other rows and record
            # the error message in the 'error' column instead of losing the row
            errStr = f"{type(e).__name__}: {e}"
            print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - {i}: ask failed: {errStr}")
        finally:
            if gate is not None:
                gate.release()
        if controller is not None:
            controller.record(metrics["ttft_s"])

        print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - {i}: after Ask (ttft_s={metrics['ttft_s']}, in_flight={in_flight})")
        # extend Parse answer from prompt.py        

        resDict = pollutedDf.iloc[i].to_dict()
        resDict.update({'prompt':promptStr,'answer':res,'error':errStr,'modelname':modelname,'promptId':promptIdVal,'expectationLiId':expectationLiId,
                        'modelConfig':modelConfigName,'reasoning':reasoningTag,
                        'ttft_s':metrics['ttft_s'],'total_time_s':metrics['total_time_s'],
                        'out_tokens':metrics['tokens'],'out_tokens_per_s':metrics['tokens_per_s'],
                        'in_flight':in_flight})
        resLi.append(resDict)
        

def getReasoningTag(reasoning_config):
    # Short human-readable tag for the reasoning setting of a model config.
    # Used in the result file name and in the 'reasoning' result column.
    if not reasoning_config:
        return "default"
    tagParts = []
    enable_thinking = reasoning_config.get("enable_thinking")
    if enable_thinking is not None and str(enable_thinking).lower() not in ("", "default"):
        tagParts.append("think_on" if bool(enable_thinking) else "think_off")
    effort = reasoning_config.get("reasoning_effort")
    if effort is not None and str(effort).lower() not in ("", "default"):
        tagParts.append(f"effort_{effort}")
    if not tagParts:
        return "default"
    return "_".join(tagParts)

def main_RuleGen():
    config_folder = Path(__file__).resolve().parent / "config"
    parameter_config_filename = config_folder / "parameter.json"
    with open(parameter_config_filename, encoding="utf-8") as param_file:
        parameter_config = json.load(param_file)

    # A run can cover several model/reasoning settings (names from connection.json).
    # Each setting is tested over the entire dataset. If "modelConfigs" is missing
    # or empty, the single "modelConfigName" is used (previous behavior).
    modelConfigLi = parameter_config.get("modelConfigs") or [parameter_config["modelConfigName"]]

    # load llm connection config
    connection_config_filename = config_folder / "connection.json"
    with open(connection_config_filename, encoding="utf-8") as config_file:
        llm_connection_config = json.load(config_file)

    #polluted_adressPerson_500_new_20251205
    # clean data
    # trainGenData_adressPerson_500_new_1.csv
    paramInitKey = parameter_config["location"]
    fileIdxArr = parameter_config[paramInitKey]["File"]["Input"]

    # prompt / expectation ids and the optional row limit are parameterised
    # (missing keys fall back to the previously hard-coded values)
    promptIdLi = parameter_config.get("promptIds") or [11,8,9,10]
    expectationLiVec = parameter_config.get("expectationLiIds") or [0,2]
    globalMaxRows = parameter_config.get("maxRows") or None
    adaptive_cfg = parameter_config.get("adaptive") or {}
    adaptive_enabled = bool(adaptive_cfg.get("enabled", True))

    for modelConfigName in modelConfigLi:
        if modelConfigName not in llm_connection_config:
            print(f"No connection config found for modelConfigName '{modelConfigName}' - skipping")
            continue

        llm_package=llm_connection_config[modelConfigName]["llm-package"]
        api_key=llm_connection_config[modelConfigName]["api_key"]
        base_url=llm_connection_config[modelConfigName]["base_url"]
        modelname=llm_connection_config[modelConfigName]["modelname"]
        reasoning_config=llm_connection_config[modelConfigName].get("reasoning") or None
        reasoningTag=getReasoningTag(reasoning_config)

        #modelname = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"#"casperhansen/llama-3.3-70b-instruct-awq"
        #modelname = "cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
        # Connection to LLM
        print(f'Connect to Client - {modelConfigName} ({reasoningTag})')
        client = getClient(llm_package,api_key,base_url)
        client
        print(f"Model: {client.models.list()}")

        # Adaptive concurrency: one gate + controller per model config, so the
        # learned thread limit applies to the whole config run (all files,
        # prompts and expectation ids of this config).
        if adaptive_enabled:
            min_threads = int(adaptive_cfg.get("min_threads", 10))
            max_threads = int(adaptive_cfg.get("max_threads", 50))
            gate = AdaptiveGate(limit=min_threads,#max_threads,
                                min_limit=min_threads,
                                max_limit=max_threads)
            controller = AdaptiveController(gate,
                                            window=int(adaptive_cfg.get("ttft_window", 20)),
                                            ttft_high=float(adaptive_cfg.get("ttft_decrease_if_above_s", 5.0)),
                                            ttft_low=float(adaptive_cfg.get("ttft_increase_if_below_s", 2.0)),
                                            decrease_factor=float(adaptive_cfg.get("decrease_factor", 0.5)),
                                            increase_step=int(adaptive_cfg.get("increase_step", 2)),
                                            cooldown_s=float(adaptive_cfg.get("cooldown_s", 15.0)))
            print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - adaptive concurrency: start with {gate.limit} threads, "
                  f"decrease if TTFT median > {adaptive_cfg.get('ttft_decrease_if_above_s', 5.0)}s, "
                  f"min {adaptive_cfg.get('min_threads', 10)}")
        else:
            gate = None
            controller = None
            print("adaptive concurrency disabled - fixed thread behaviour as before")

        for fileIdx in fileIdxArr:
            inputFileParam = parameter_config["Files"][fileIdx]
            data_folder = Path(parameter_config[paramInitKey]["Paths"]["Data"])
            print("----------------------------------------------------------")
            print(f"START file {inputFileParam['name']}")
            csvDf_dirty = pd.read_csv(data_folder/inputFileParam['name'],
                                encoding=(None if inputFileParam['encoding'] in "None" else inputFileParam['encoding']),
                                delimiter=(None if inputFileParam['delimiter'] in "None" else inputFileParam['delimiter']))
            # optional row limit for small test runs: per-file "maxRows" overrides the top-level one
            maxRows = inputFileParam.get("maxRows")
            if maxRows is None:
                maxRows = globalMaxRows
            if maxRows:
                n_rows_before = len(csvDf_dirty)
                csvDf_dirty = csvDf_dirty.head(int(maxRows))
                print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - maxRows: testing first {len(csvDf_dirty)} of {n_rows_before} rows of {inputFileParam['name']}")

            csvDf_clean = pd.read_csv(data_folder/inputFileParam['cleanFile'],
                                encoding=(None if inputFileParam['encoding'] in "None" else inputFileParam['encoding']),
                                delimiter=(None if inputFileParam['delimiter'] in "None" else inputFileParam['delimiter']),
                                dtype='object')
            csvDf_clean['pk'] = csvDf_clean['pk'].apply(lambda x: int(x))

            if 'n_cols' in inputFileParam.keys():
                n_cols = inputFileParam['n_cols']
            else:
                n_cols = 14
            if 'id_column' in inputFileParam.keys():
                id_column = inputFileParam['id_column']
            else:
                id_column = 'pk'
            if 'err_dect_info' in inputFileParam.keys():
                err_dect_info = inputFileParam['err_dect_info']
            else:
                err_dect_info = None

            print('Start Reading')

            result_folder = Path(parameter_config[paramInitKey]["Paths"]["Result"])
            resFileSuff = parameter_config["Files"][fileIdx]["OutputSuffix"]

            df_fil = csvDf_dirty.query("isDirty == True")
            # df_fil = csvDf_dirty.iloc[:, 1:]
            # df_fil = df_new.query('";spellingMistake" == errRule')
            file_content = None
            # promptIdLi / expectationLiVec come from parameter.json (see top of main_RuleGen)
            #promptIdVal = 10 #9
            for promptIdVal in promptIdLi:
                resLi = list()
                modeVal = 'w'
                expectationLiId = 0
                for expectationLiId in expectationLiVec:
                    isHeader = True
                    offsetStr = f"_{modelConfigName}_{reasoningTag}_{promptIdVal}_{expectationLiId}_{datetime.now().strftime('%y-%m-%d_%H-%M-%S')}"
                    pollutedDf = df_fil
                    threadNum = 52 # 26 or 52
                    threads = [None] * threadNum #52
                    i = 0
                    print(f"len: {len(csvDf_dirty)}")

                    for index, row in csvDf_dirty.iterrows():
                        print('##################################')
                        print(f'row: {index}')
                        #def askModelSaveAnswer(llm_package,client,modelname,promptIdVal,
                        #           cleanDf,pollutedDf,file_content,resLi,i)
                        #n_cols=14,id_column='pk',err_dect_info=None
                        threads[i] = threading.Thread(target=askModelSaveAnswer, 
                                args=(llm_package,client,modelname,promptIdVal,
                                    csvDf_clean,pollutedDf,file_content,resLi,index,expectationLiId,
                                    n_cols,id_column,err_dect_info,
                                    reasoning_config,modelConfigName,reasoningTag,
                                    gate,controller))
                        threads[i].start()

                        i=i+1
                        if (index % (threadNum-3)) == 0 and index > 1:
                            print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - index: {index}")
                            for j in range(len(threads)):
                                # print(f"j in loop: {j}")
                                if threads[j] is None:
                                    print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - j None: {j}")
                                    break
                                print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - j before join: {j}")
                                threads[j].join()
                                print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - j after join: {j}")
                            dfAll = pd.DataFrame(resLi)
                            dfAll = dfAll.assign(modelname=modelname)
                            print(f"save file resDf_all_{offsetStr}_{resFileSuff}.csv")
                            print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - {index}: before to_csv")
                            dfAll.to_csv(result_folder/f'resDf_all_{offsetStr}_{resFileSuff}.csv', mode=modeVal,header=isHeader,index=False)
                            print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - {index}: after to_csv")
                            resLi = []
                            isHeader = False
                            modeVal = 'a'
                            print(len(threads))
                            threads = [None] * len(threads)
                            i=0

                        #if index==5:
                        #    break

                    for j in range(len(threads)):
                        if threads[j] is None or i == 0:
                            break
                        threads[j].join()

                    # get the list of tuples from two lists.
                    # and merge them by using zip().
                    #list_of_tuples = list(zip(idxLi, strLi, hnLi, stLi, tnLi, plzLi, ortLi, prLi, repLi))
                    #dfAll = createResDf(list_of_tuples,['idx', 'Strasse', 'Hausnr', 'Stiege', 'Tuernr', 'Plz', 'Ort', 'Prompt', 'Responds'])
                    # 'Prompt', 'PromptId'
                    dfAll = pd.DataFrame(resLi)

                    print(dfAll)
                    fileName = f'resDf_all_{offsetStr}_{resFileSuff}.csv'
                    print(f"fin save file {fileName}")
                    dfAll = dfAll.assign(modelname=modelname)
                    dfAll.to_csv(result_folder/fileName, mode=modeVal,header=isHeader,index=False)
        if controller is not None:
            print(f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')} - [ADAPT] final status {modelConfigName}: {controller.status()}")
    print('-------------- fin all --------------')


if __name__ == "__main__":
    main_RuleGen()
