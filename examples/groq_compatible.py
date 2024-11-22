import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
from groq import Groq
from FlagEmbedding import BGEM3FlagModel

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

# Initialize BGE-M3 model
bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
#bge_model = BGEM3FlagModel('BAAI/bge-m3-unsupervised', use_fp16=True)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    print("in llm_model_func")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=messages,
        **kwargs
    )
    return response.choices[0].message.content

async def embedding_func(texts: list[str]) -> np.ndarray:
    embeddings = bge_model.encode(texts, batch_size=12, max_length=8192)['dense_vecs']
    print(f"Actual embedding dimension: {embeddings.shape[1]}")
    return np.array(embeddings)

async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    
    return embedding.shape[1]

async def main():
    try:
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")
    except Exception as e:
        print(f"Error detecting embedding dimension: {e}")
        return

    try:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,  # Use detected dimension
                max_token_size=8192,
                func=embedding_func,
            ),
        )
    except Exception as e:
        print(f"Error initializing LightRAG: {e}")
        return

    try:
        print("kiscica")
        with open("./text.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())
    except FileNotFoundError:
        print("Error: text.txt file not found.")
        return
    except Exception as e:
        print(f"Error inserting text into RAG: {e}")
        return

    search_modes = ["naive", "local", "global", "hybrid"]
    for mode in search_modes:
        try:
            result = await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode=mode)
            )
            print(f"\nResults for {mode} search:")
            print(result)
        except Exception as e:
            print(f"Error performing {mode} search: {e}")

if __name__ == "__main__":
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())