# PDF Summarizer with Llama 3.2, LangChain, and ChromaDB

A Python application that uses RAG (Retrieval-Augmented Generation) to summarize PDF documents and answer questions about their content using Llama 3.2, LangChain, and ChromaDB.

## Features

- Load and process PDF documents
- Generate multiple types of summaries (comprehensive, brief, bullet points, abstract)
- Ask custom questions about PDF content
- Extract key topics and themes
- Uses RAG for accurate, context-aware responses
- Vector storage with ChromaDB for efficient retrieval



## Installation


1. Install required Python packages:

```bash
pip install langchain langchain-ollama langchain-community langchain-chroma pypdf chromadb
```

2. Install and set up Ollama:

```bash
# Install Ollama from https://ollama.ai/

# Pull the Llama 3.2 model
ollama pull llama3.2
```

## Project Structure

```
project/
│
├── PdfSummarization.py    # Main application file
├── requirements.txt       # 📦 Python dependencies
└── README.md              # 📖 This file
└── documents/             # Folder for your PDF files
    └── your_document.pdf
```

## Setup

1. Create a `documents` folder in the same directory as the script:

```bash
mkdir documents
```

2. Place your PDF files in the `documents` folder

## Usage

Run the script:

```bash
python PdfSummarization.py
```

The application will:
1. List all PDF files found in the `documents` folder
2. Allow you to select which PDF to process
3. Generate multiple types of summaries:
   - Comprehensive summary
   - Brief summary
   - Key topics extraction
   - Answer to a custom question

### Available Summary Types

The `summarize()` method supports different summary types:

- `"comprehensive"` - Detailed summary covering all main topics
- `"brief"` - Short 2-3 paragraph summary
- `"bullet_points"` - 7-10 bullet points of main ideas
- `"abstract"` - Academic-style abstract

### Custom Questions

You can modify the script to ask your own questions:

```python
answer = summarizer.ask_question("What methodology was used in this research?")
print(answer)
```
