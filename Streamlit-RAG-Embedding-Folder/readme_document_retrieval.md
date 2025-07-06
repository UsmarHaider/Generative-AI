# 📚 Document Retrieval Chat with Gemini + FAISS + Streamlit

This project lets you **upload documents (PDF or TXT)**, automatically split them into chunks, create **embeddings** using **Google Gemini Embeddings**, and **search inside your documents** using **FAISS** for fast retrieval.  
It also allows you to **chat with your documents** — Gemini AI will answer your questions based on document content.

---

## 🚀 Features
- 📄 Upload PDF and TXT documents
- ✂️ Automatic text chunking using LangChain
- 🔢 Embedding generation with **Google Gemini Embeddings**
- ⚡ FAISS-based vector database for fast similarity search
- 💬 Ask questions and get answers from documents
- 📅 Save and Load embeddings locally for reuse
- 🎨 Built with **Streamlit** for an interactive UI

---

## 📂 Folder Structure
```
saved_embeddings/
│
├── my_documents.index  # FAISS vector index (embeddings)
└── my_documents.pkl    # Pickle file containing document metadata
```

---

## 📥 Installation

1. **Clone the repo:**
```bash
git clone https://github.com/UsmarHaider/Generative-AI/tree/main/Streamlit-RAG
cd Streamlit-RAG
```

2. **Create virtual environment & activate it:**
```bash
python -m venv .venv
# For Windows:
.venv\Scripts\activate
# For Mac/Linux:
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set Google API Key:**
- Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- Either:
  - Export it:
    ```bash
    export GOOGLE_API_KEY=your-api-key
    ```
  - Or enter it inside the app sidebar.

---

## 🏃‍♂️ Run The App:
```bash
streamlit run app.py
```

---

## ✅ How To Use:
1. Enter your **Google API Key** inside sidebar.
2. Upload PDF or TXT documents.
3. Click "**Process Documents**".
4. Search by asking questions in chat interface.
5. You can:
   - Save embeddings using "**Save Embeddings to Folder**".
   - Load them later using "**Load Saved Embeddings**" (no need to reprocess).

---

## 🧐 Tech Stack:
- **Streamlit** (Frontend UI)
- **FAISS** (Fast Vector Search)
- **LangChain** (Text Splitting)
- **Google Gemini AI** (Embeddings + Generative Answering)
- **PyPDF2** (PDF Reading)
- **Pickle** (Saving Documents Info)

---

## 📜 License:
MIT License

---

## 🤝 Contributing:
PRs and Issues are welcome!  
Feel free to fork and improve the app.

---

## 💡 Credits:
- Google Generative AI
- LangChain
- FAISS
- Streamlit

---

## 📞 Contact:
Created by **Usmar Haider**  
Connect on [LinkedIn](https://www.linkedin.com/in/usmarhaider/)

