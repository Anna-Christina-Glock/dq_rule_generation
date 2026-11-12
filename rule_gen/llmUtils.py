import time

import pandas as pd
import ollamaModels as ollamaMod
import openAIModels as openAiMod


def getClient(llm_package,api_key,base_url):    
    if llm_package == "ollama":
        client = ollamaMod.getClient(base_url)
    elif llm_package == "openai":
        client = openAiMod.getClient(api_key,base_url)
    else:
        print("no llm package registered, no matching Client found")
        client = None
    return client

def askModel(llm_package,client,promptStr,modelname=None,reasoning_config=None):  
    if llm_package == "ollama":
        # reasoning settings are only forwarded for openai-compatible endpoints for now
        resStr = ollamaMod.askLLM(promptStr,client,modelname)
    elif llm_package == "openai":
        resStr = openAiMod.askLLM(promptStr,client,modelname,reasoning_config=reasoning_config)
    elif llm_package == "huggingFace":
        resStr = openAiMod.askLLM(promptStr,client,modelname,reasoning_config=reasoning_config)
    else:
        print("no llm package registered, no matching Client found")
        resStr = None
    return resStr

def askModelWithMetrics(llm_package,client,promptStr,modelname=None,reasoning_config=None):
    # Like askModel, but returns (resStr, metrics):
    #   metrics["ttft_s"]       seconds to the first streamed chunk (None on the
    #                           non-streaming ollama path - no TTFT there)
    #   metrics["total_time_s"] seconds for the whole request
    #   metrics["tokens"]       completion tokens if the server reports usage
    #   metrics["tokens_per_s"] output tokens per second after the first token
    t_start = time.monotonic()
    if llm_package == "ollama":
        # reasoning settings are only forwarded for openai-compatible endpoints for now
        resStr = ollamaMod.askLLM(promptStr,client,modelname)
        metrics = {"ttft_s":None,"total_time_s":time.monotonic()-t_start,
                   "tokens":None,"tokens_per_s":None}
    elif llm_package == "openai":
        resStr, metrics = openAiMod.askLLMStream(promptStr,client,modelname,reasoning_config=reasoning_config)
    elif llm_package == "huggingFace":
        resStr, metrics = openAiMod.askLLMStream(promptStr,client,modelname,reasoning_config=reasoning_config)
    else:
        print("no llm package registered, no matching Client found")
        resStr = None
        metrics = {"ttft_s":None,"total_time_s":time.monotonic()-t_start,
                   "tokens":None,"tokens_per_s":None}
    return resStr, metrics

def createResDf(list_of_tuples, colnames):
    # Converting lists of tuples into
    # pandas Dataframe.
    dfAll = pd.DataFrame(list_of_tuples,
                    columns=colnames)
    return dfAll