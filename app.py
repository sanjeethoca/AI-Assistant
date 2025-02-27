import streamlit as st
import os
import duckdb
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ✅ Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Connect to DuckDB Database
DB_PATH = "document_embeddings.duckdb"
con = duckdb.connect(DB_PATH)

# ✅ Ensure table exists
con.execute("""
CREATE TABLE IF NOT EXISTS document_embeddings (
    id TEXT PRIMARY KEY,
    file_name TEXT,
    chunk TEXT,
    embedding BLOB
);
""")

# ✅ Function to Query DuckDB for Relevant Documents
def query_duckdb(query, model, top_n=3):
    """
    Retrieve relevant document chunks from DuckDB using cosine similarity.
    """
    query_embedding = model.encode([query])[0]  # Generate embedding for query

    # Fetch all stored embeddings from DuckDB
    stored_data = con.execute("SELECT id, file_name, chunk, embedding FROM document_embeddings").fetchall()

    results = []
    for row in stored_data:
        chunk_id, file_name, chunk_text, stored_embedding = row

        # Convert binary to numpy array
        stored_embedding = np.frombuffer(stored_embedding, dtype=np.float32)

        # Compute cosine similarity
        similarity = np.dot(query_embedding, stored_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
        
        results.append((chunk_text, file_name, similarity))

    # Sort by similarity and get top N results
    results = sorted(results, key=lambda x: x[2], reverse=True)[:top_n]

    # Format retrieved context
    retrieved_chunks = "\n\n".join([f"📄 **From {file}**:\n{text}" for text, file, _ in results])

    return retrieved_chunks if retrieved_chunks else "No relevant context found."

# ✅ Groq API Configuration
GROQ_API_KEY = "your_groq_api_key_here"  # 🔹 Replace with your actual API Key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ✅ Function to Call Groq's Llama 3.3 - 70B Model
def ask_groq_llm(prompt, model="llama-3.3-70b-versatile", max_tokens=500):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a highly accurate AI assistant, limited to answering based only on the provided context."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3  # 🔹 Lower value for more factual responses
    }

    response = requests.post(GROQ_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# ✅ Function to Generate RAG-based Answers
def generate_rag_response(user_query):
    """
    Retrieve relevant context from DuckDB and generate a response using Llama 3.3 - 70B.
    """
    
    # Step 1: Retrieve document chunks from DuckDB
    context = query_duckdb(user_query, model)

    # Step 2: Construct a precise RAG prompt
    prompt = f"""
    You are OCA Assistant, an AI trained on Offshore Construction Associates (OCA) documents.
    Act as an expert with the information you know from the documents. Give answers as if you're a consultant. 
    Your job is to answer based ONLY on the provided document information.
    If the requested information is missing, reply:
    "I don’t have enough information to answer that."
    DO not give responses such as: "Based on the provided OCA document"

    📄 **Context from OCA Documents:**
    {context}

    ❓ **User Question:** {user_query}

    💡 **Answer:**
    """

    # Step 3: Call Groq’s Llama 3.3 - 70B to generate an answer
    response = ask_groq_llm(prompt, model="llama-3.3-70b-versatile")

    return response

# ✅ Streamlit UI
st.title("OCA AI Assistant")

user_input = st.text_input("Ask a question about OCA:")
if st.button("Get Answer"):
    if user_input:
        with st.spinner("Generating response..."):
            response = generate_rag_response(user_input)
            st.write(response)
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.info("OCA-BOT")
