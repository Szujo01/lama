import os
from openai import OpenAI
from lightrag import LightRAG

# Configure OpenAI client to use Groq
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Initialize LightRAG
light_rag = LightRAG()

# Function to perform RAG query
def rag_query(query):
    # Use LightRAG to retrieve relevant documents
    relevant_docs = light_rag.retrieve(query)
    
    # Prepare context from retrieved documents
    context = "\n".join(relevant_docs)
    
    # Prepare messages for the chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the user's question."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    
    # Generate response using Groq (via OpenAI-compatible API)
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # Assuming Groq supports this model
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Example usage
query = "What is the capital of France?"
answer = rag_query(query)
print(f"Question: {query}")
print(f"Answer: {answer}")