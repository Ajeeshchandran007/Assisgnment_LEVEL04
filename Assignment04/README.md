# LLM-MongoDB RAG System

A Retrieval-Augmented Generation (RAG) system that combines LangChain, OpenAI, and local MongoDB for intelligent document querying. This system works with **local MongoDB instances** without requiring MongoDB Atlas.

## Features

- 🚀 **Local MongoDB Support** - No cloud dependencies required
- 📚 **Multiple File Format Support** - PDF, DOCX, DOC, TXT, MD, JSON, and more
- 🔍 **Semantic Search** - Vector similarity search with OpenAI embeddings
- 💬 **Interactive Query Mode** - Chat-style interface for document Q&A
- 🌊 **Streaming Support** - Real-time response streaming
- 📊 **Collection Statistics** - Monitor your document database
- 🔧 **LangChain 1.0+ Compatible** - Uses latest LCEL patterns

## Requirements

 **Start MongoDB:**
   ```bash
   # macOS/Linux
   mongod --dbpath /path/to/your/data/directory
   
   # Or use brew services on macOS
   brew services start mongodb-community
   ```

5. **Create a `.env` file** in the project directory:
   ```env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   MONGO_URI=mongodb://localhost:27017/
   ```

   Get your OpenAI API key from: https://platform.openai.com/api-keys

## Project Structure

```
.
├── LLM_MongoDB.py          # Main application code
├── .env                    # Environment variables (create this)
├── documents/              # Place your documents here (create this)
│   ├── file1.pdf
│   ├── file2.docx
│   └── file3.txt
└── README.md               # This file
```

## Usage

### Basic Usage

1. **Create a `documents` folder** and add your files:
   ```bash
   mkdir documents
   # Copy your PDF, DOCX, TXT files into this folder
   ```

2. **Run the application:**
   ```bash
   python LLM_MongoDB.py
   ```

3. **Ask questions about your documents:**
   ```
   🤔 Your question: What are the main topics covered in these documents?
   ```

### Interactive Commands

- **Ask questions** - Type any question about your documents
- **`stats`** - View collection statistics
- **`clear`** - Clear all documents from the database
- **`exit`** or **`quit`** - Exit the application



## Supported File Formats

- **Text Files:** `.txt`, `.md`, `.json`, `.py`, `.js`, `.html`, `.css`
- **PDF Files:** `.pdf` (requires PyPDF2)
- **Word Documents:** `.docx` (requires python-docx), `.doc` (requires textract)



## How It Works

1. **Document Loading** - Reads files from the `documents` folder
2. **Text Splitting** - Breaks documents into manageable chunks
3. **Embedding Generation** - Creates vector embeddings using OpenAI
4. **Storage** - Stores embeddings and text in local MongoDB
5. **Retrieval** - Performs similarity search to find relevant chunks
6. **Generation** - Uses GPT-4o-mini to generate answers from retrieved context
