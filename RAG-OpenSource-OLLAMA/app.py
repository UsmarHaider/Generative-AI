#RAG
import os
import re
import json
import string
import requests
import streamlit as st
import faiss
import numpy as np
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Constants
EMBEDDING_DIM = 384  # for 'all-MiniLM-L6-v2'
CHUNK_SIZE = 500

# Initialize model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(EMBEDDING_DIM)
metadata_store = []

# Text extraction
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    return ""

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Cleaning function
def clean_text(text):
    text = ''.join(filter(lambda x: x in string.printable, text))  # Keep only printable chars
    text = re.sub(r'\s+', ' ', text)  # Collapse all whitespace
    return text.strip()

# Chunking
def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Embedding + Storage
def store_embeddings(chunks):
    global index, metadata_store
    embeddings = model.encode(chunks)
    embeddings = normalize(embeddings)
    index.add(np.array(embeddings, dtype=np.float32))
    metadata_store.extend(chunks)

# Retrieval
def search_chunks(query, k=5):
    embedding = model.encode([query])
    embedding = normalize(embedding)
    embedding = np.array(embedding, dtype=np.float32)
    D, I = index.search(embedding, k)
    return [metadata_store[i] for i in I[0]]

# Ollama generation
def generate_answer_with_ollama(query, context):
    prompt = f"""You are an AI assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "deepseek-r1", "prompt": prompt},
        stream=True
    )

    answer = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            answer += data.get("response", "")
    return answer

# Streamlit app
def main():
    st.set_page_config(page_title="RAG Chatbot with Ollama", layout="wide")
    st.title("📚 RAG Chatbot using FAISS + Ollama (deepseek-r1)")

    os.makedirs("uploaded_files", exist_ok=True)
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        all_chunks = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join("uploaded_files", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            raw_text = extract_text(file_path)
            cleaned = clean_text(raw_text)
            chunks = chunk_text(cleaned)
            all_chunks.extend(chunks)

        if all_chunks:
            store_embeddings(all_chunks)
            st.success("✅ Documents processed and embeddings stored.")

    query = st.text_input("🔍 Ask something about the uploaded documents:")

    if query:
        if len(metadata_store) == 0:
            st.warning("⚠️ Please upload and process documents first.")
        else:
            relevant_chunks = search_chunks(query, k=5)
            context = "\n\n".join(relevant_chunks)
            st.info("Top matching sections:")
            for chunk in relevant_chunks:
                st.write(f"- {chunk[:300]}...")

            with st.spinner("💬 Generating response from DeepSeek..."):
                answer = (query, context)
                st.success("💡 Answer:")
                st.write(answer)

if __name__ == "__main__":
    main()
