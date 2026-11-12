from ollama import Client


def getClient(base_url):
    return Client(host=base_url)


def askLLM(promptStr, client, modelname="mistral:7b"):
    response = client.chat(
        model=modelname,
        messages=[{"role": "user", "content": promptStr}],
    )
    return response["message"]["content"]
