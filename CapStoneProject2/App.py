"""
RAG System - Streamlit Web Application (LangChain 1.0 - Simple Approach)
=========================================================================
Interactive web interface for querying documents using local Ollama LLM

Requirements:
pip install streamlit langchain-community langchain-chroma langchain-text-splitters pypdf langchain-ollama

Run with: streamlit run App.py
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage


# Page configuration
st.set_page_config(
    page_title="Local RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="stSidebar"] h2 {
        color: white !important;
    }
    [data-testid="stSidebar"] p {
        color: rgba(255, 255, 255, 0.9) !important;
    }
</style>
""", unsafe_allow_html=True)


# Configuration
class Config:
    MODEL_NAME = "llama3.2"
    EMBEDDING_MODEL = "nomic-embed-text"
    CHROMA_PERSIST_DIR = "./chroma_db"
    DOCUMENTS_DIR = "./documents"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    RETRIEVAL_K = 5
    TEMPERATURE = 0.1


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📂 Document Upload"


def initialize_models():
    """Initialize Ollama models"""
    try:
        with st.spinner("🔄 Initializing Ollama models..."):
            # Initialize embeddings
            st.session_state.embeddings = OllamaEmbeddings(
                model=Config.EMBEDDING_MODEL
            )
            
            # Initialize LLM
            st.session_state.llm = ChatOllama(
                model=Config.MODEL_NAME,
                temperature=Config.TEMPERATURE,
                num_ctx=4096
            )
            
            st.session_state.initialized = True
            st.session_state.current_page = "📂 Document Upload"
            return True
    except Exception as e:
        st.error(f"❌ Failed to initialize models: {e}")
        st.info("💡 Make sure Ollama is running and models are pulled:\n"
                f"- `ollama pull {Config.MODEL_NAME}`\n"
                f"- `ollama pull {Config.EMBEDDING_MODEL}`")
        return False


def load_document(file_path):
    """Load a single document"""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Unsupported file format")
    
    return loader.load()


def create_vectorstore(documents):
    """Create vector store from documents"""
    try:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=st.session_state.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )
        
        return vectorstore, len(chunks)
    except Exception as e:
        st.error(f"❌ Error creating vectorstore: {e}")
        return None, 0


def load_existing_vectorstore():
    """Load existing vector store"""
    try:
        vectorstore = Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR,
            embedding_function=st.session_state.embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error loading vectorstore: {e}")
        return None


def query_documents(question):
    """Query documents using simple retrieval + LLM approach"""
    try:
        # Retrieve relevant documents
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        
        relevant_docs = retriever.invoke(question)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt
        system_message = SystemMessage(content="""You are a helpful assistant. Use the following context to answer the question.

Instructions:
- Read the context carefully and extract the relevant information
- If the answer is in the context, provide it clearly and directly
- If you cannot find the answer in the context, say "I don't know based on the provided documents"
- Be specific and cite relevant parts from the context""")
        
        human_message = HumanMessage(content=f"""Context:
{context}

Question: {question}

Answer:""")
        
        # Get response from LLM
        messages = [system_message, human_message]
        response = st.session_state.llm.invoke(messages)
        
        # Return result
        return {
            'answer': response.content,
            'source_documents': relevant_docs
        }
        
    except Exception as e:
        st.error(f"❌ Error querying documents: {e}")
        return None


def sidebar_navigation():
    """Sidebar navigation"""
    st.sidebar.markdown("## 🎛️ Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["📂 Document Upload", "🔍 Query System", "📊 Query History"],
        index=["📂 Document Upload", "🔍 Query System", "📊 Query History"].index(st.session_state.current_page),
        label_visibility="collapsed"
    )
    
    # Update current page
    st.session_state.current_page = page
    
    st.sidebar.markdown("---")
    
    # Status indicators
    st.sidebar.markdown("## 📊 System Status")
    
    if st.session_state.initialized:
        st.sidebar.success("✅ Models Initialized")
    else:
        st.sidebar.warning("⚠️ Models Not Initialized")
    
    if st.session_state.vectorstore is not None:
        st.sidebar.success("✅ Documents Loaded")
    else:
        st.sidebar.info("ℹ️ No Documents Loaded")
    
    st.sidebar.markdown("---")
    
    # Configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    st.sidebar.text(f"Model: {Config.MODEL_NAME}")
    st.sidebar.text(f"Embeddings: {Config.EMBEDDING_MODEL}")
    st.sidebar.text(f"Chunks: {Config.RETRIEVAL_K}")
    
    return page


def document_upload_page():
    """Document upload page"""
    st.markdown('<div class="main-header">📂 Document Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload documents or load existing database</div>', unsafe_allow_html=True)
    
    # Initialize models
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", type="primary"):
            if initialize_models():
                st.success("✅ System initialized successfully!")
                st.balloons()
                st.info("⏳ Redirecting to Document Upload in 5 seconds...")
                import time
                time.sleep(5)
                st.rerun()
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["📤 Upload Files", "💾 Load Existing"])
    
    with tab1:
        st.markdown("### Upload PDF or TXT Files")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Select one or more PDF or TXT files"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            
            # Show file details
            with st.expander("📋 View Selected Files"):
                for i, file in enumerate(uploaded_files, 1):
                    file_size = len(file.getvalue()) / 1024
                    st.write(f"{i}. **{file.name}** ({file_size:.2f} KB)")
            
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                process_uploaded_files(uploaded_files)
    
    with tab2:
        st.markdown("### Load Existing Vector Database")
        st.info(f"📁 Database location: `{Config.CHROMA_PERSIST_DIR}`")
        
        if os.path.exists(Config.CHROMA_PERSIST_DIR) and os.listdir(Config.CHROMA_PERSIST_DIR):
            st.success("✅ Existing database found!")
            
            if st.button("📥 Load Database", type="primary", use_container_width=True):
                with st.spinner("🔄 Loading database..."):
                    vectorstore = load_existing_vectorstore()
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.success("✅ Database loaded successfully!")
                        st.balloons()
        else:
            st.warning("⚠️ No existing database found")
            st.info("💡 Upload documents first using the Upload tab")


def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    with st.spinner("🔄 Processing documents..."):
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                all_docs = []
                
                # Process each file
                for uploaded_file in uploaded_files:
                    # Save file
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load document
                    docs = load_document(file_path)
                    all_docs.extend(docs)
                
                # Create vectorstore
                if all_docs:
                    vectorstore, num_chunks = create_vectorstore(all_docs)
                    
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.success(f"✅ Processed {len(all_docs)} pages into {num_chunks} chunks!")
                        st.balloons()
                else:
                    st.warning("⚠️ No documents were loaded")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")


def query_page():
    """Query page"""
    st.markdown('<div class="main-header">🔍 Query System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your documents</div>', unsafe_allow_html=True)
    
    # Check initialization
    if not st.session_state.initialized:
        st.warning("⚠️ System not initialized")
        st.info("💡 Go to Document Upload page to initialize")
        return
    
    if st.session_state.vectorstore is None:
        st.warning("⚠️ No documents loaded!")
        st.info("💡 Upload documents in the Document Upload page first")
        return
    
    # Query input
    st.markdown("### ❓ Ask Your Question")
    question = st.text_area(
        "Enter your question",
        height=100,
        placeholder="What would you like to know about the documents?"
    )
    
    # Submit button
    if st.button("🚀 Submit Query", type="primary", use_container_width=True):
        if not question.strip():
            st.error("❌ Please enter a question")
            return
        
        process_query(question)


def process_query(question):
    """Process query and display results"""
    with st.spinner("🤖 Generating answer..."):
        try:
            # Query the system
            result = query_documents(question)
            
            if result is None:
                return
            
            # Display results
            st.markdown("---")
            st.markdown("## 📋 Results")
            
            # Answer
            st.markdown("### 💡 Answer")
            st.info(result['answer'])
            
            # Sources
            with st.expander("📚 View Sources", expanded=False):
                for i, doc in enumerate(result['source_documents'], 1):
                    source_file = doc.metadata.get('source', 'Unknown')
                    if '\\' in source_file or '/' in source_file:
                        source_file = os.path.basename(source_file)
                    
                    st.markdown(f"**[{i}] {source_file}** (Page: {doc.metadata.get('page', 'N/A')})")
                    content_preview = doc.page_content[:300].replace('\n', ' ')
                    st.caption(content_preview + "...")
                    st.markdown("---")
            
            # Add to history
            st.session_state.query_history.append({
                'question': question,
                'answer': result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.success("✅ Query completed!")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())


def history_page():
    """Query history page"""
    st.markdown('<div class="main-header">📊 Query History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">View your recent queries</div>', unsafe_allow_html=True)
    
    if not st.session_state.query_history:
        st.info("📭 No queries yet. Ask a question first!")
        return
    
    st.markdown(f"### 📝 Recent Queries ({len(st.session_state.query_history)})")
    
    # Display history in reverse order (most recent first)
    for i, item in enumerate(reversed(st.session_state.query_history), 1):
        with st.expander(f"Query {i}: {item['question'][:50]}..."):
            st.markdown(f"**🕐 Time:** {item['timestamp']}")
            st.markdown(f"**❓ Question:** {item['question']}")
            st.markdown(f"**💡 Answer Preview:** {item['answer']}")
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.query_history = []
        st.rerun()


def main():
    """Main application"""
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    page = sidebar_navigation()
    
    # Route to appropriate page
    if page == "📂 Document Upload":
        document_upload_page()
    elif page == "🔍 Query System":
        query_page()
    elif page == "📊 Query History":
        history_page()


if __name__ == "__main__":
    main()