"""
RAG Agent - Streamlit Web Application
=====================================
Interactive web interface for RAG agent with file upload and notifications.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from datetime import datetime
import sqlite3

try:
    from RagAgent import RAGAgent
    from config import Config
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.info("Please ensure RagAgent.py and config.py are in the same directory.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    p.main-header {
        font-size: 3.5rem !important;
        font-weight: bold !important;
        color: #667eea !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
        line-height: 1.2 !important;
        display: block !important;
    }
    p.sub-header {
        font-size: 1.5rem !important;
        color: #6c757d !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        line-height: 1.4 !important;
        display: block !important;
    }
    /* Override Streamlit's default paragraph styling */
    [data-testid="stMarkdownContainer"] p.main-header {
        font-size: 3.5rem !important;
    }
    [data-testid="stMarkdownContainer"] p.sub-header {
        font-size: 1.5rem !important;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .notification-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    /* Sidebar navigation styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 1.1rem;
        font-weight: 600;
        color: white !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        background-color: transparent !important;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        color: white !important;
        font-size: 1rem;
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
        transform: translateX(5px);
    }
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background-color: rgba(255, 255, 255, 0.3) !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'vectorstore_loaded' not in st.session_state:
        st.session_state.vectorstore_loaded = False
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []


def initialize_agent():
    """Initialize the RAG agent"""
    try:
        if st.session_state.agent is None:
            with st.spinner("🔄 Initializing RAG Agent..."):
                st.session_state.agent = RAGAgent(Config)
            return True
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG Agent: {e}")
        st.info("💡 Check your OpenAI API key in config.py")
        return False
    return True


def save_uploaded_file(uploaded_file, temp_dir):
    """Save uploaded file to temporary directory"""
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_pasted_content(content, content_type, temp_dir):
    """Save pasted content as a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = "txt" if content_type == "Text" else "pdf"
    file_path = os.path.join(temp_dir, f"pasted_content_{timestamp}.{file_ext}")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return file_path


def sidebar_navigation():
    """Sidebar navigation and configuration"""
    st.sidebar.markdown("## 🎛️ Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["📂 Document Upload", "🔍 Query System", "📊 View History", "⚙️ Configuration"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Status indicators
    st.sidebar.markdown("## 📊 System Status")
    
    if st.session_state.agent is not None:
        st.sidebar.success("✅ Agent Initialized")
    else:
        st.sidebar.warning("⚠️ Agent Not Initialized")
    
    if st.session_state.vectorstore_loaded:
        st.sidebar.success("✅ Vector Store Loaded")
    else:
        st.sidebar.info("ℹ️ No Documents Loaded")
    
    st.sidebar.markdown("---")
    
    # Quick stats
    if st.session_state.agent is not None:
        try:
            conn = sqlite3.connect(st.session_state.agent.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM summaries")
            query_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM document_metadata")
            doc_count = cursor.fetchone()[0]
            
            conn.close()
            
            st.sidebar.markdown("## 📈 Statistics")
            st.sidebar.metric("Total Queries", query_count)
            st.sidebar.metric("Documents Processed", doc_count)
        except:
            pass
    
    return page


def document_upload_page():
    """Document upload and processing page"""
    st.markdown('<p class="main-header">📂 Document Management</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload or paste documents to process</p>', unsafe_allow_html=True)
    
    # Initialize agent if not done
    if not initialize_agent():
        return
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📤 Upload Files", "💾 Load Existing"])
    
    with tab1:
        st.markdown("### Upload PDF or TXT Files")
        st.markdown("Upload multiple files at once. Supported formats: PDF, TXT")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Select one or more PDF or TXT files to upload"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            
            # Show file details
            with st.expander("📋 View Selected Files"):
                for i, file in enumerate(uploaded_files, 1):
                    file_size = len(file.getvalue()) / 1024  # KB
                    st.write(f"{i}. **{file.name}** ({file_size:.2f} KB)")
            
            if st.button("🚀 Process Uploaded Files", type="primary", use_container_width=True):
                process_uploaded_files(uploaded_files)
    
    
    
    with tab2:
        st.markdown("### Load Existing Vector Database")
        st.info(f"📁 Vector database location: `{Config.CHROMA_PERSIST_DIRECTORY}`")
        
        if os.path.exists(Config.CHROMA_PERSIST_DIRECTORY):
            st.success("✅ Existing vector database found!")
            
            if st.button("📥 Load Vector Database", type="primary", use_container_width=True):
                load_existing_database()
        else:
            st.warning("⚠️ No existing vector database found")
            st.info("💡 Process some documents first using the Upload or Paste tabs")


def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    with st.spinner("🔄 Processing documents..."):
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save all uploaded files
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file, temp_dir)
                    file_paths.append(file_path)
                
                # Separate PDFs and TXTs
                pdf_files = [f for f in file_paths if f.endswith('.pdf')]
                txt_files = [f for f in file_paths if f.endswith('.txt')]
                
                # Process documents
                from langchain.schema import Document
                all_docs = []
                
                # Load PDFs
                if pdf_files:
                    st.info(f"📄 Loading {len(pdf_files)} PDF file(s)...")
                    pdf_docs = st.session_state.agent.load_pdf_documents(pdf_files)
                    all_docs.extend(pdf_docs)
                
                # Load TXTs
                if txt_files:
                    st.info(f"📝 Loading {len(txt_files)} TXT file(s)...")
                    for txt_file in txt_files:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            doc = Document(
                                page_content=content,
                                metadata={"source": os.path.basename(txt_file)}
                            )
                            all_docs.append(doc)
                
                # Process and store
                if all_docs:
                    success = st.session_state.agent.process_and_store_documents(documents=all_docs)
                    
                    if success:
                        st.session_state.vectorstore_loaded = True
                        st.session_state.processing_complete = True
                        st.success(f"✅ Successfully processed {len(all_docs)} document(s)!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to process documents")
                else:
                    st.warning("⚠️ No documents were loaded")
        
        except Exception as e:
            st.error(f"❌ Error processing files: {e}")


def process_pasted_text(content, name, content_type):
    """Process pasted text content"""
    with st.spinner("🔄 Processing pasted content..."):
        try:
            from langchain.schema import Document
            
            doc = Document(
                page_content=content,
                metadata={"source": f"{name}.txt"}
            )
            
            success = st.session_state.agent.process_and_store_documents(documents=[doc])
            
            if success:
                st.session_state.vectorstore_loaded = True
                st.session_state.processing_complete = True
                st.success(f"✅ Successfully processed pasted content!")
                st.balloons()
            else:
                st.error("❌ Failed to process content")
        
        except Exception as e:
            st.error(f"❌ Error processing content: {e}")


def load_existing_database():
    """Load existing vector database"""
    with st.spinner("🔄 Loading vector database..."):
        try:
            success = st.session_state.agent.load_existing_vectorstore()
            
            if success:
                st.session_state.vectorstore_loaded = True
                st.success("✅ Vector database loaded successfully!")
            else:
                st.error("❌ Failed to load vector database")
        
        except Exception as e:
            st.error(f"❌ Error loading database: {e}")


def query_page():
    """Query and notification page"""
    st.markdown('<p class="main-header">🔍 Query System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions and send notifications</p>', unsafe_allow_html=True)
    
    # Check if agent is initialized
    if st.session_state.agent is None:
        st.warning("⚠️ Please initialize the agent first")
        if st.button("🔄 Initialize Agent"):
            initialize_agent()
        return
    
    # Check if vector store is loaded
    if not st.session_state.vectorstore_loaded:
        st.warning("⚠️ No documents loaded!")
        st.info("💡 Please upload documents in the Document Upload page first")
        return
    
    # Query input
    st.markdown("### ❓ Ask Your Question")
    question = st.text_area(
        "Enter your question",
        height=100,
        placeholder="What would you like to know about the documents?"
    )
    
    # Notification options
    st.markdown("### 📤 Notification Options")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        send_all = st.checkbox("📢 Send ALL", value=False, help="Send to all channels")
    
    with col2:
        send_whatsapp = st.checkbox("💬 WhatsApp", value=False, disabled=send_all)
    
    with col3:
        send_sms = st.checkbox("📱 SMS", value=False, disabled=send_all)
    
    with col4:
        send_email = st.checkbox("📧 Email", value=False, disabled=send_all)
    
    with col5:
        send_twitter = st.checkbox("🐦 Twitter", value=False, disabled=send_all)
    
    # Override individual selections if "Send ALL" is checked
    if send_all:
        send_whatsapp = send_sms = send_email = send_twitter = True
    
    # Process query
    if st.button("🚀 Submit Query", type="primary", use_container_width=True):
        if not question.strip():
            st.error("❌ Please enter a question")
            return
        
        process_query(question, send_whatsapp, send_sms, send_email, send_twitter)


def process_query(question, send_whatsapp, send_sms, send_email, send_twitter):
    """Process query and send notifications"""
    with st.spinner("🤖 Generating answer..."):
        try:
            # Determine if any notifications are selected
            any_notifications = any([send_whatsapp, send_sms, send_email, send_twitter])
            
            if any_notifications:
                result = st.session_state.agent.process_query_with_notifications(
                    question=question,
                    send_whatsapp=send_whatsapp,
                    send_sms=send_sms,
                    send_email=send_email,
                    send_twitter=send_twitter
                )
            else:
                result = st.session_state.agent.query(question)
                result['notifications'] = {}
            
            # Display results
            st.markdown("---")
            st.markdown("## 📋 Query Results")
            
            # Summary
            st.markdown("### 💡 Summary")
            st.info(result['summary'])
            
            # Full Answer
            with st.expander("📄 View Full Answer", expanded=True):
                st.markdown(result['answer'])
            
            # Sources
            with st.expander("📚 View Sources"):
                for i, source in enumerate(result['sources'], 1):
                    st.markdown(f"**{i}. {os.path.basename(source['source'])}** (Page: {source['page']})")
                    st.caption(source['content_preview'])
                    st.markdown("---")
            
            # Notification Status
            if any_notifications:
                st.markdown("### 📤 Notification Status")
                
                notification_results = result.get('notifications', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if send_whatsapp:
                        if notification_results.get('whatsapp', False):
                            st.success("✅ WhatsApp")
                        else:
                            st.error("❌ WhatsApp")
                
                with col2:
                    if send_sms:
                        if notification_results.get('sms', False):
                            st.success("✅ SMS")
                        else:
                            st.error("❌ SMS")
                
                with col3:
                    if send_email:
                        if notification_results.get('email', False):
                            st.success("✅ Email")
                        else:
                            st.error("❌ Email")
                
                with col4:
                    if send_twitter:
                        if notification_results.get('twitter', False):
                            st.success("✅ Twitter")
                        else:
                            st.error("❌ Twitter")
            
            # Add to history
            st.session_state.query_history.append({
                'question': question,
                'summary': result['summary'],
                'timestamp': result['timestamp']
            })
            
            st.success("✅ Query processed successfully!")
        
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")


def history_page():
    """View query history and summaries"""
    st.markdown('<p class="main-header">📊 Query History</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">View your recent queries and document metadata</p>', unsafe_allow_html=True)
    
    if st.session_state.agent is None:
        st.warning("⚠️ Agent not initialized")
        return
    
    tab1, tab2 = st.tabs(["📝 Query Summaries", "📚 Document Metadata"])
    
    with tab1:
        st.markdown("### Recent Query Summaries")
        
        limit = st.slider("Number of summaries to display", 5, 50, 10)
        
        if st.button("🔄 Refresh Summaries"):
            summaries = st.session_state.agent.get_summaries(limit=limit)
            
            if summaries:
                for i, summary in enumerate(summaries, 1):
                    with st.expander(f"Query {i}: {summary['query'][:50]}..."):
                        st.markdown(f"**🕒 Timestamp:** {summary['timestamp']}")
                        st.markdown(f"**❓ Query:** {summary['query']}")
                        st.markdown(f"**💡 Summary:**")
                        st.info(summary['summary'])
            else:
                st.info("📭 No summaries found. Process some queries first!")
    
    with tab2:
        st.markdown("### Processed Documents")
        
        if st.button("🔄 Refresh Metadata"):
            try:
                conn = sqlite3.connect(st.session_state.agent.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_path, file_type, pages, processed_date 
                    FROM document_metadata 
                    ORDER BY processed_date DESC
                """)
                results = cursor.fetchall()
                conn.close()
                
                if results:
                    for i, (path, file_type, pages, date) in enumerate(results, 1):
                        with st.container():
                            col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
                            with col1:
                                st.markdown(f"**{os.path.basename(path)}**")
                            with col2:
                                st.badge(file_type)
                            with col3:
                                st.write(f"{pages} pages")
                            with col4:
                                st.caption(date)
                            st.markdown("---")
                else:
                    st.info("📭 No documents processed yet")
            
            except Exception as e:
                st.error(f"❌ Error loading metadata: {e}")


def configuration_page():
    """Configuration and system status page"""
    st.markdown('<p class="main-header">⚙️ Configuration</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">System configuration and API settings</p>', unsafe_allow_html=True)
    
    # System Status
    st.markdown("### 🔧 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("OpenAI Model", Config.OPENAI_MODEL)
        st.metric("Embedding Model", Config.EMBEDDING_MODEL)
        st.metric("Chunk Size", Config.CHUNK_SIZE)
    
    with col2:
        st.metric("Temperature", Config.TEMPERATURE)
        st.metric("Retrieval K", Config.RETRIEVAL_K)
        st.metric("Chunk Overlap", Config.CHUNK_OVERLAP)
    
    st.markdown("---")
    
    # API Configuration Status
    st.markdown("### 🔑 API Configuration")
    
    api_configs = {
        "OpenAI API": bool(Config.OPENAI_API_KEY and not Config.OPENAI_API_KEY.startswith("your-")),
        "Twilio (WhatsApp/SMS)": bool(Config.TWILIO_ACCOUNT_SID and not Config.TWILIO_ACCOUNT_SID.startswith("your-")),
        "Email SMTP": bool(Config.EMAIL_SENDER and Config.EMAIL_PASSWORD),
        "Twitter API": bool(Config.TWITTER_API_KEY and not Config.TWITTER_API_KEY.startswith("your-"))
    }
    
    for service, configured in api_configs.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{service}**")
        with col2:
            if configured:
                st.success("✅ Configured")
            else:
                st.error("❌ Not Configured")
    
    st.markdown("---")
    
    # File Paths
    st.markdown("### 📁 Storage Locations")
    
    st.text_input("Vector Database", Config.CHROMA_PERSIST_DIRECTORY, disabled=True)
    st.text_input("SQLite Database", Config.SQLITE_DB_PATH, disabled=True)
    st.text_input("Documents Directory", Config.DOCUMENTS_DIRECTORY, disabled=True)
    
    st.markdown("---")
    
    # Notification Configuration
    st.markdown("### 📤 Notification Recipients")
    
    st.text_input("WhatsApp", Config.RECIPIENT_WHATSAPP, disabled=True)
    st.text_input("SMS", Config.RECIPIENT_PHONE, disabled=True)
    st.text_input("Email", Config.EMAIL_RECIPIENT, disabled=True)
    
    st.info("💡 To modify configuration, edit the `config.py` file and restart the app")


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
    elif page == "📊 View History":
        history_page()
    elif page == "⚙️ Configuration":
        configuration_page()


if __name__ == "__main__":
    main()