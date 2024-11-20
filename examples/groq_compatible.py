import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
import httpx
from ollama import Client as OllamaClient

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def groq_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": "mixtral-8x7b-32768",  # You can change this to other Groq models
        "messages": messages,
        **kwargs
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await groq_complete(prompt, system_prompt, history_messages, **kwargs)

async def bge_m3_embedding(texts: list[str]) -> np.ndarray:
    client = OllamaClient()
    embeddings = []
    for text in texts:
        response = await client.embeddings(model='bge-m3', prompt=text)
        embeddings.append(response['embedding'])
    return np.array(embeddings)

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await bge_m3_embedding(texts)

async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = len(embedding[0])
    return embedding_dim

# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)

# asyncio.run(test_funcs())

async def main():
    try:
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=8192,
                func=embedding_func,
            ),
        )
        # Rest of your main function...
    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    asyncio.run(main())