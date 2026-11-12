
import time

from openai import OpenAI

def getClient(api_key,base_url):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return client

def buildReasoningKwargs(reasoning_config):
    # Translate the optional "reasoning" setting of a connection.json entry into
    # kwargs for client.chat.completions.create():
    #   reasoning_config = None / {}                     -> model default, nothing sent
    #   {"enable_thinking": true|false}                  -> extra_body["chat_template_kwargs"]["enable_thinking"]
    #   {"reasoning_effort": "low"|"medium"|"high"}      -> extra_body["reasoning_effort"]
    # Both are sent via extra_body (merged into the top level of the request JSON),
    # which is what a vLLM server reads. This also works with openai SDK versions
    # that do not expose reasoning_effort as a named parameter of create().
    extra_body = {}
    if reasoning_config:
        effort = reasoning_config.get("reasoning_effort")
        if effort is not None and str(effort).lower() not in ("", "default"):
            extra_body["reasoning_effort"] = effort
        enable_thinking = reasoning_config.get("enable_thinking")
        if enable_thinking is not None and str(enable_thinking).lower() not in ("", "default"):
            extra_body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
    if extra_body:
        return {"extra_body": extra_body}
    return {}

def askLLM(promptStr, client, modelname, picPath=None, reasoning_config=None):
    model = modelname # client.models.list().data[0].id
    if picPath is None:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                    "role": "user",
                    "content": promptStr
                }
            ],
            **buildReasoningKwargs(reasoning_config),
        )
    else:
        messages = [{
                "role": "user",
                "content": {
                    "content": "Hello, how are you?",
                    "files": [
                        {
                            "type": "image",
                            "path": 'C:\\Users\\glock\\Downloads\\002_smalldatajourney-1.png'
                        }
                    ]
                }
            }
        ]

    resStr = response.to_dict()["choices"][0]["message"]["content"]
    return resStr

def askLLMStream(promptStr, client, modelname, reasoning_config=None):
    """Streaming counterpart of askLLM: collects the answer while measuring latency.

    Returns (resStr, metrics) with
        metrics["ttft_s"]       - seconds from request sent until the first
                                  streamed chunk (i.e. queue wait + prefill).
                                  This is the congestion signal the adaptive
                                  controller uses. None if no chunk arrived.
        metrics["total_time_s"] - seconds from request sent until the last chunk
        metrics["tokens"]       - completion tokens reported by the server
                                  (via stream_options include_usage), else None
        metrics["tokens_per_s"] - output tokens / generation time (after TTFT),
                                  if both are known, else None

    For reasoning models the first chunks are thinking tokens - that is
    intended: TTFT to the first chunk marks the end of prefill/queueing,
    which is what matters for KV-cache pressure. (TTFT to the first *visible*
    token would be contaminated by thinking time.)
    """
    t_start = time.monotonic()
    first_chunk_ts = None
    chunks = client.chat.completions.create(
        model=modelname,
        messages=[{
                "role": "user",
                "content": promptStr
            }
        ],
        stream=True,
        stream_options={"include_usage": True},
        **buildReasoningKwargs(reasoning_config),
    )
    parts = []
    completion_tokens = None
    for chunk in chunks:
        if first_chunk_ts is None:
            first_chunk_ts = time.monotonic()
        if getattr(chunk, "usage", None) is not None:
            completion_tokens = getattr(chunk.usage, "completion_tokens", None)
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta is not None and delta.content:
                parts.append(delta.content)
    t_end = time.monotonic()
    resStr = "".join(parts)
    ttft_s = (first_chunk_ts - t_start) if first_chunk_ts is not None else None
    metrics = {
        "ttft_s": ttft_s,
        "total_time_s": t_end - t_start,
        "tokens": completion_tokens,
        "tokens_per_s": None,
    }
    if completion_tokens and ttft_s is not None and t_end > first_chunk_ts:
        metrics["tokens_per_s"] = completion_tokens / (t_end - first_chunk_ts)
    return resStr, metrics
