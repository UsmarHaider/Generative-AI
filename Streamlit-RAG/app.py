import streamlit as st
import os
import tempfile
import pickle
from typing import List, Dict, Any
import faiss
import numpy as np
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Document Retrieval Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DocumentProcessor:
    """Handles document processing and text extraction"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, txt_file) -> str:
        """Extract text from TXT file"""
        try:
            return txt_file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading text file: {str(e)}")
            return ""
    
    def process_documents(self, uploaded_files) -> List[Document]:
        """Process uploaded files and return list of Document objects"""
        documents = []
        
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = self.extract_text_from_pdf(file)
            elif file.type == "text/plain":
                text = self.extract_text_from_txt(file)
            else:
                st.warning(f"Unsupported file type: {file.type}")
                continue
            
            if text:
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file.name,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
        
        return documents

class VectorDatabase:
    """Manages Faiss vector database operations"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []
        self.dimension = None
    
    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Create embeddings for documents"""
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        return np.array(embeddings).astype('float32')
    
    def build_index(self, documents: List[Document]):
        """Build Faiss index from documents"""
        if not documents:
            st.error("No documents to index")
            return
        
        with st.spinner("Creating embeddings..."):
            embeddings = self.create_embeddings(documents)
            self.dimension = embeddings.shape[1]
            
            # Create Faiss index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.index.add(embeddings)
            self.documents = documents
            
        st.success(f"Successfully indexed {len(documents)} document chunks")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None:
            return []
        
        # Create query embedding
        query_embedding = np.array([self.embedding_model.embed_query(query)]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(score),
                    "rank": i + 1
                })
        
        return results
    
    def save_index(self, filepath: str):
        """Save Faiss index and documents to disk"""
        if self.index is not None:
            # Save Faiss index
            faiss.write_index(self.index, f"{filepath}.index")
            
            # Save documents and metadata
            with open(f"{filepath}.pkl", "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "dimension": self.dimension
                }, f)
    
    def load_index(self, filepath: str):
        """Load Faiss index and documents from disk"""
        try:
            # Load Faiss index
            self.index = faiss.read_index(f"{filepath}.index")
            
            # Load documents and metadata
            with open(f"{filepath}.pkl", "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.dimension = data["dimension"]
            
            return True
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")
            return False

def initialize_session_state():
    """Initialize session state variables"""
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False

def setup_api_key():
    """Setup and validate Google API key"""
    st.sidebar.header("🔑 API Configuration")
    
    # Check for API key in environment or user input
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        api_key = st.sidebar.text_input(
            "Enter your Google API Key",
            type="password",
            help="Get your API key from Google AI Studio"
        )
    
    if api_key:
        try:
            # Set environment variable
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
            
            # Test the API key
            model = genai.GenerativeModel('gemini-1.5-flash')
            model.generate_content("Hello")
            
            st.session_state.api_key_valid = True
            st.sidebar.success("✅ API Key validated successfully!")
            return True
        except Exception as e:
            st.sidebar.error(f"❌ Invalid API Key: {str(e)}")
            st.session_state.api_key_valid = False
            return False
    else:
        st.sidebar.warning("⚠️ Please enter your Google API Key to continue")
        return False

def main():
    """Main application function"""
    st.title("📚 Document Retrieval Chat")
    st.markdown("Upload documents and chat with them using Google's Gemini embeddings!")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup API key
    if not setup_api_key():
        st.stop()
    
    # Initialize components
    doc_processor = DocumentProcessor()
    
    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF or TXT files to create a searchable knowledge base"
        )
        
        if uploaded_files:
            st.write(f"📄 {len(uploaded_files)} file(s) uploaded")
            
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Process documents
                    documents = doc_processor.process_documents(uploaded_files)
                    
                    if documents:
                        # Initialize embedding model
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/embedding-001",
                            task_type="RETRIEVAL_DOCUMENT"
                        )
                        
                        # Initialize vector database
                        vector_db = VectorDatabase(embeddings)
                        vector_db.build_index(documents)
                        
                        # Store in session state
                        st.session_state.vector_db = vector_db
                        st.session_state.documents_processed = True
                        
                        st.success("✅ Documents processed successfully!")
                    else:
                        st.error("❌ No valid documents found")
        
        # Display processing status
        if st.session_state.documents_processed:
            st.success("📊 Vector database ready!")
            if st.session_state.vector_db:
                st.info(f"📚 {len(st.session_state.vector_db.documents)} chunks indexed")
    
    # Main chat interface
    if st.session_state.documents_processed and st.session_state.vector_db:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.expander(f"💭 Question {i+1}: {question[:50]}..."):
                st.write("**Question:**", question)
                st.write("**Answer:**", answer)
        
        # Chat input
        question = st.text_input("Ask a question about your documents:", key="question_input")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        
        with col2:
            num_results = st.slider("Number of results", 1, 10, 3)
        
        if search_button and question:
            with st.spinner("Searching..."):
                # Search for relevant documents
                results = st.session_state.vector_db.search(question, k=num_results)
                
                if results:
                    # Display search results
                    st.subheader("🔍 Search Results")
                    
                    context_text = ""
                    for result in results:
                        doc = result["document"]
                        score = result["score"]
                        
                        with st.expander(f"📄 Result {result['rank']} - {doc.metadata['source']} (Score: {score:.3f})"):
                            st.write("**Content:**")
                            st.write(doc.page_content)
                            st.write("**Metadata:**")
                            st.json(doc.metadata)
                        
                        context_text += f"\n\n{doc.page_content}"
                    
                    # Generate answer using Gemini
                    try:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        prompt = f"""
                        Based on the following context from the uploaded documents, please answer the question.
                        
                        Context:
                        {context_text}
                        
                        Question: {question}
                        
                        Please provide a comprehensive answer based on the context. If the context doesn't contain enough information to answer the question, please say so.
                        """
                        
                        response = model.generate_content(prompt)
                        answer = response.text
                        
                        # Display answer
                        st.subheader("🤖 AI Answer")
                        st.write(answer)
                        
                        # Add to chat history
                        st.session_state.chat_history.append((question, answer))
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                
                else:
                    st.warning("No relevant documents found for your question.")
    
    else:
        st.info("👆 Please upload and process documents in the sidebar to start chatting!")
        
        # Display instructions
        st.markdown("""
        ## How to use:
        
        1. **Add your Google API Key** in the sidebar
        2. **Upload documents** (PDF or TXT files)
        3. **Click "Process Documents"** to create embeddings
        4. **Ask questions** about your documents in the chat interface
        
        ## Features:
        - 📄 Support for PDF and TXT files
        - 🔍 Semantic search using Google's Gemini embeddings
        - 💾 Efficient vector storage with Faiss
        - 🤖 AI-powered answers using Gemini Pro
        - 📊 Multiple search results with relevance scores
        """)

if __name__ == "__main__":
    main()