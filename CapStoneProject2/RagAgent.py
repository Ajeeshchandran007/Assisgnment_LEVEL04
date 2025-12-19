"""
RAG System with Local Ollama LLM and ChromaDB - LangChain 1.0+ Version
Uses modern LangChain architecture with LCEL (LangChain Expression Language)

Requirements:
pip install langchain langchain-community langchain-chroma langchain-ollama pypdf

Make sure Ollama is running with required models:
ollama pull llama3.2
ollama pull nomic-embed-text
"""

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os


class LocalRAGSystem:
    def __init__(self, 
                 model_name="llama3.2", 
                 embedding_model="nomic-embed-text",
                 persist_directory="./chroma_db"):
        """
        Initialize RAG system with local Ollama LLM and ChromaDB using LangChain 1.0+
        
        Args:
            model_name: Ollama LLM model (e.g., 'llama3.2', 'llama2', 'mistral')
            embedding_model: Ollama embedding model (e.g., 'nomic-embed-text', 'mxbai-embed-large')
            persist_directory: Directory to store ChromaDB embeddings
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        
        # Initialize embeddings using Ollama
        print(f"Initializing Ollama embeddings with model: {embedding_model}")
        print("Make sure you have run: ollama pull nomic-embed-text")
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Initialize Ollama LLM with better settings
        print(f"Connecting to Ollama with model: {model_name}")
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            num_ctx=4096
        )
        
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
    
    def load_document(self, file_path):
        """
        Load document (PDF or TXT) and return documents
        
        Args:
            file_path: Path to the document file
        """
        print(f"Loading document: {file_path}")
        
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError("Unsupported file format. Use .pdf or .txt")
        
        documents = loader.load()
        print(f"Loaded {len(documents)} pages/documents")
        return documents
    
    def load_documents_from_directory(self, directory_path="./documents"):
        """
        Load all PDF and TXT documents from a directory
        
        Args:
            directory_path: Path to the directory containing documents
        """
        print(f"Loading documents from directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        all_documents = []
        
        # Load PDF files
        try:
            pdf_loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            pdf_docs = pdf_loader.load()
            all_documents.extend(pdf_docs)
            print(f"Loaded {len(pdf_docs)} PDF pages")
        except Exception as e:
            print(f"No PDF files found or error loading PDFs: {e}")
        
        # Load TXT files
        try:
            txt_loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                show_progress=True
            )
            txt_docs = txt_loader.load()
            all_documents.extend(txt_docs)
            print(f"Loaded {len(txt_docs)} TXT documents")
        except Exception as e:
            print(f"No TXT files found or error loading TXTs: {e}")
        
        if not all_documents:
            raise ValueError(f"No documents found in {directory_path}")
        
        print(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def create_embeddings(self, documents):
        """
        Split documents into chunks and create embeddings in ChromaDB
        
        Args:
            documents: List of documents to embed
        """
        print("\nSplitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        print("\nCreating embeddings and storing in ChromaDB...")
        print("This may take a few minutes depending on document size...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("✓ Embeddings created and stored successfully!")
    
    def load_existing_vectorstore(self):
        """
        Load existing ChromaDB vectorstore from disk
        """
        print(f"Loading existing vectorstore from {self.persist_directory}")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print("✓ Vectorstore loaded successfully!")
    
    def format_docs(self, docs):
        """
        Format retrieved documents into a single string
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def setup_rag_chain(self):
        """
        Setup the RAG chain using LCEL (LangChain Expression Language)
        This is the modern LangChain 1.0+ approach
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Load or create embeddings first.")
        
        # Setup retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create prompt template using ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Use the following context to answer the question.

Instructions:
- Read the context carefully and extract the relevant information
- If the answer is in the context, provide it clearly and directly
- If you cannot find the answer in the context, say "I don't know based on the provided documents"
- Be specific and cite relevant parts from the context"""),
            ("human", """Context:
{context}

Question: {question}

Answer:""")
        ])
        
        # Build RAG chain using LCEL
        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✓ RAG chain setup complete using LCEL!")
    
    def query(self, question):
        """
        Query the RAG system
        
        Args:
            question: User's question
        
        Returns:
            dict with 'result' and 'source_documents'
        """
        if self.rag_chain is None:
            raise ValueError("RAG chain not setup. Run setup_rag_chain() first.")
        
        print(f"\n🔍 Processing query: {question}")
        
        # Get the answer
        answer = self.rag_chain.invoke(question)
        
        # Get source documents separately
        source_docs = self.retriever.invoke(question)
        
        return {
            'result': answer,
            'source_documents': source_docs
        }


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("LOCAL RAG SYSTEM - LangChain 1.0+ with LCEL")
    print("="*70)
    
    # Initialize RAG system
    print("\n📋 Initializing RAG system...")
    rag = LocalRAGSystem(
        model_name="llama3.2",
        embedding_model="nomic-embed-text"
    )
    
    documents_dir = "./documents"
    
    # Check if we should create new embeddings or load existing ones
    if os.path.exists(rag.persist_directory) and os.listdir(rag.persist_directory):
        print("\n✓ Found existing embeddings database")
        print("\nOptions:")
        print("  1. Use existing embeddings (fast)")
        print("  2. Recreate embeddings from documents (slow)")
        user_choice = input("\nYour choice [1/2]: ").strip()
        
        if user_choice == "2":
            print("\n📁 Loading and Processing Documents...")
            documents = rag.load_documents_from_directory(documents_dir)
            rag.create_embeddings(documents)
        else:
            print("\n📂 Loading Existing Embeddings...")
            rag.load_existing_vectorstore()
    else:
        print(f"\n📁 No existing embeddings found")
        if not os.path.exists(documents_dir):
            print(f"Creating directory: {documents_dir}")
            os.makedirs(documents_dir)
            print(f"\n⚠️  Please add your PDF or TXT files to the '{documents_dir}' folder")
            print("   and run the script again.")
            exit(0)
        
        print("\n📄 Loading and Processing Documents...")
        documents = rag.load_documents_from_directory(documents_dir)
        rag.create_embeddings(documents)
    
    # Setup RAG chain
    print("\n⚙️  Setting up RAG chain...")
    rag.setup_rag_chain()
    
    # Interactive query loop
    print("\n" + "="*70)
    print("🚀 RAG SYSTEM READY!")
    print("="*70)
    print("\nYou can now ask questions about your documents.")
    print("Type 'quit', 'exit', or 'q' to stop.\n")
    print("-"*70)
    
    while True:
        user_query = input("\n💬 Your question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_query:
            continue
        
        try:
            response = rag.query(user_query)
            
            print("\n" + "="*70)
            print("📝 ANSWER:")
            print("="*70)
            print(response['result'])
            
            print("\n" + "-"*70)
            print("📚 SOURCE DOCUMENTS:")
            print("-"*70)
            for i, doc in enumerate(response['source_documents'], 1):
                source_file = doc.metadata.get('source', 'Unknown')
                # Extract just the filename
                if '\\' in source_file or '/' in source_file:
                    source_file = os.path.basename(source_file)
                
                print(f"\n[{i}] Source: {source_file}")
                print(f"    Page: {doc.metadata.get('page', 'N/A')}")
                content_preview = doc.page_content[:250].replace('\n', ' ')
                print(f"    Content: {content_preview}...")
            
            print("\n" + "="*70)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()