# 🤖 Local RAG System with Ollama & ChromaDB

A Retrieval-Augmented Generation (RAG) system that runs completely locally on your desktop using **Llama 3.2** via Ollama, ChromaDB for vector storage, and LangChain for orchestration.

## 📋 Overview

This system allows you to:
- Load private PDF/TXT documents
- Convert them to embeddings using local models
- Store embeddings in a local ChromaDB vector database
- Query your documents using natural language
- Get answers powered by **Llama 3.2** running locally on your machine

---

## 🛠️ Prerequisites


### Pull Required Ollama Models

Download the **Llama 3.2** model and embedding model:

```bash
# Pull Llama 3.2 (LLM for answering questions)
ollama pull llama3.2

# Pull Nomic Embed Text (for creating embeddings)
ollama pull nomic-embed-text
```


You should see both `llama3.2` and `nomic-embed-text` in the list.

---

## 📁 Project Structure

```
rag-system/
│
├── RagAgent.py          # Main CLI application
├── App.py               # Streamlit web interface
├── config.py            # Configuration settings
├── documents/           # Place your PDF/TXT files here
├── chroma_db/           # Vector database (created automatically)
└── README.md            # This file
```

---

## 🎯 Usage Guide

### Option 1: Command Line Interface (Recommended for First Run)

#### Step 1: Prepare Your Documents

Create a `documents` folder and add your PDF or TXT files:

```bash
mkdir documents
# Copy your files into the documents/ folder
```

#### Step 2: Run the RAG System

```bash
python RagAgent.py
```

#### Step 3: Initialize the System

On first run, the system will:
1. Load all documents from the `documents/` folder
2. Split them into chunks
3. Create embeddings using `nomic-embed-text`
4. Store embeddings in ChromaDB (`chroma_db/` folder)

**This process may take a few minutes depending on document size.**

#### Step 4: Query Your Documents

Once initialized, you'll see:
```
🚀 RAG SYSTEM READY!
======================================================================

You can now ask questions about your documents.
Type 'quit', 'exit', or 'q' to stop.

💬 Your question:
```

**Example queries:**
```
💬 Your question: What is the main topic of the document?
💬 Your question: Summarize the key points about project management
💬 Your question: What does the document say about deadlines?
```

#### Step 5: View Results

The system will show:
- ✅ **Answer** from Llama 3.2
- 📚 **Source documents** with page numbers and excerpts

---

### Option 2: Web Interface (Streamlit)

For a visual, user-friendly interface:

```bash
streamlit run App.py
```

This will open a web browser with:
- 📂 **Document Upload** page
- 🔍 **Query System** page
- 📊 **Query History** page

---

## ⚙️ Configuration

Edit `config.py` to customize settings:

```python
class Config:
    # Models
    OLLAMA_MODEL = "llama3.2"           # LLM model
    EMBEDDING_MODEL = "nomic-embed-text" # Embedding model
    
    # Document Processing
    CHUNK_SIZE = 1500                     # Size of text chunks
    CHUNK_OVERLAP = 300                  # Overlap between chunks
    RETRIEVAL_K = 5                      # Number of chunks to retrieve
    
    # Directories
    DOCUMENTS_DIRECTORY = "./documents"
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
```



## 📝 Important Notes

### First Run vs Subsequent Runs

**First Run:**
- Creates embeddings (slow, 2-10 minutes depending on document size)
- Stores in `chroma_db/` folder

**Subsequent Runs:**
- Loads existing embeddings (fast, <5 seconds)
- No need to reprocess documents

### Adding New Documents

To add new documents:

1. Place new files in `documents/` folder
2. Run the application
3. Choose option `2` to recreate embeddings

Or delete the `chroma_db/` folder to rebuild from scratch:
```bash
rm -rf chroma_db/
python RagAgent.py
```


## 🎓 How It Works

1. **Document Loading**: PDFs and text files are loaded using LangChain loaders
2. **Text Splitting**: Documents are split into smaller chunks with overlap
3. **Embedding Generation**: Each chunk is converted to a vector using `nomic-embed-text`
4. **Vector Storage**: Embeddings are stored in ChromaDB (local database)
5. **Query Processing**: 
   - User question is embedded
   - Similar chunks are retrieved from ChromaDB
   - Chunks are sent to Llama 3.2 with the question
   - Llama 3.2 generates an answer based on the context

---

## 📚 Tech Stack

- **LLM**: Llama 3.2 (via Ollama)
- **Embeddings**: Nomic Embed Text
- **Vector DB**: ChromaDB
- **Framework**: LangChain 1.0+
- **Document Processing**: PyPDF, LangChain loaders
- **UI**: Streamlit (optional)

---

