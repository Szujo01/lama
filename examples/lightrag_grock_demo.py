import os
import asyncio
from groq import AsyncGroq
from lightrag import LightRAG, QueryParam
from dotenv import load_dotenv

load_dotenv()

# Set up AsyncGroq client
async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

async def groq_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    try:
        response = await async_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in Groq API call: {e}")
        return ""

async def groq_complete_main(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await groq_complete(prompt, system_prompt, history_messages, **kwargs)

# Synchronous wrapper for groq_complete_main
def sync_groq_complete_main(prompt, system_prompt=None, history_messages=[], **kwargs):
    return asyncio.run(groq_complete_main(prompt, system_prompt, history_messages, **kwargs))

# Set up LightRAG with Groq
WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=sync_groq_complete_main,
    llm_model_name="llama3-8b-8192",
    tiktoken_model_name="gpt-4",  # Use a compatible tokenizer
)

# Rest of your code remains the same
with open("./book.txt", "r", encoding="UTF-16") as f:
    rag.insert(f.read())

# Perform searches
print(rag.query("Mi az a alapvető eszköz?", param=QueryParam(mode="naive")))
print(rag.query("Mi az a fogyasztói vezeték?", param=QueryParam(mode="local")))
print(rag.query("Mi vonatkozik az irányításra?", param=QueryParam(mode="global")))
print(rag.query("Mit jelent a lakossági fogyasztó?", param=QueryParam(mode="hybrid")))