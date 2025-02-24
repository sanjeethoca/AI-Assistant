import streamlit as st
import chromadb
import requests
import os
import sys
import sqlite3

# Ensure ChromaDB uses the correct SQLite version
sys.modules["pysqlite3"] = sqlite3
sys.modules["sqlite3"] = sqlite3

# ✅ Ensure persistent ChromaDB connection
DB_PATH = os.path.abspath(r"c:\Users\AmitPrajapati\Desktop\AI_BOT\chroma_db")  # Get absolute path

if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient(path=DB_PATH)

chroma_client = st.session_state.chroma_client

# ✅ Load the embeddings collection
collection_name = "document_embeddings"
try:
    collection = chroma_client.get_collection(collection_name)
    # st.success(f"✅ Connected to '{collection_name}' collection!")
except Exception:
    collection = chroma_client.create_collection(collection_name)
    st.success(f"🚀 Created and connected to '{collection_name}' collection!")

# ✅ Check Available Collections
collections = chroma_client.list_collections()
# st.write("📁 Available Collections:", [col for col in collections])

# ✅ Check if ChromaDB has data
# st.write("🔍 Checking document count in ChromaDB...")
try:
    count = collection.count()
    # st.write(f"📊 Number of documents stored: {count}")
    if count == 0:
        st.warning("⚠️ No documents found in ChromaDB! Try reloading your data.")
except Exception as e:
    st.error(f"⚠️ Error fetching document count: {e}")

# ✅ Groq API Configuration
GROQ_API_KEY = "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"  # 🔹 Replace with your actual API Key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ✅ Function to Query ChromaDB for Relevant Documents
def query_chromadb(query, top_n=3):
    """Retrieve relevant document chunks from ChromaDB."""
    try:
        query_results = collection.query(query_texts=[query], n_results=top_n)
        retrieved_chunks = query_results.get("documents", [[]])[0]
        return "\n\n".join(retrieved_chunks) if retrieved_chunks else "No relevant context found."
    except Exception as e:
        return f"❌ Error querying ChromaDB: {str(e)}"

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
        "temperature": 0.3
    }
    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        st.error(f"⚠️ API Error {response.status_code}: {response.text}")
        return "⚠️ Error fetching response from API."

# ✅ Function to Generate RAG-based Answers
def generate_rag_response(user_query):
    """Retrieve relevant context from ChromaDB and generate a response using Llama 3.3 - 70B."""
    st.write(f"📝 Querying ChromaDB with: {user_query}")
    context = query_chromadb(user_query)
    prompt = f"""
    You are OCA Assistant, an AI trained on Offshore Construction Associates (OCA) documents.
    Your job is to answer based ONLY on the provided document information.
    If the requested information is missing, reply:
    "I don’t have enough information to answer that."
    
    📄 **Context from OCA Documents:**
    {context}
    
    ❓ **User Question:** {user_query}
    
    💡 **Answer:**
    """
    return ask_groq_llm(prompt, model="llama-3.3-70b-versatile")

# ✅ Streamlit UI
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
