# 📚 RAG Agent with Multi-Channel Notifications

A production-ready Retrieval-Augmented Generation (RAG) system with PDF support, vector storage, and automated notifications via WhatsApp, SMS, Email, and Twitter.

## ✨ Features

- **📄 PDF Support**: Process PDF documents along with text files
- **🔍 Semantic Search**: ChromaDB vector database for efficient retrieval
- **🤖 OpenAI Integration**: GPT-4 for intelligent answers, text-embedding-3-small for embeddings
- **💾 SQLite Storage**: Stores query summaries and metadata locally
- **🔔 Multi-Channel Notifications**:
  - 📱 WhatsApp (via Twilio)
  - 💬 SMS (via Twilio)
  - 📧 Email (via SMTP)
  - 🐦 Twitter/X (via API v2)
- **⚙️ Flexible Configuration**: Centralized config management
- **🎯 Interactive Interface**: User-friendly menu system
- **🔒 Security**: Environment variable support

---

## 📁 Project Structure

```
Rag-agent/
├── config.py              # ⚙️ Configuration management
├── rag_agent.py           # 🤖 Main RAG agent class
├── App.py                 # 🖥️ Interactive menu interface
├── requirements.txt       # 📦 Python dependencies
├── documents/             # 📂 Place your PDFs/TXT files here (auto-created)
├── chroma_db/             # 🗄️ Vector database storage (auto-created)
├── summaries.db           # 💾 SQLite database (auto-created)
└── README.md              # 📖 This file
```

---


### Run the Streamlit App

```bash
streamlit run App.py


### Access the Web Interface

After running the command, you'll see:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

**Open your browser** and go to: `http://localhost:8501`


## 🎨 App Interface Overview

The app opens with a **beautiful gradient sidebar** and **four main pages**:

```
┌─────────────────────────────────────┐
│  🎛️ Navigation Sidebar              │
│  ├─ 📂 Document Upload              │
│  ├─ 🔍 Query System                 │
│  ├─ 📊 View History                 │
│  └─ ⚙️ Configuration                │
└─────────────────────────────────────┘
```

---

## 🔥 Features & How They Work

### Feature 1: 📂 Document Upload Page

**Purpose**: Upload and process your documents to create a searchable knowledge base

#### How It Works:

1. **Upload Files Tab**
   ```
   Click "Choose files" → Select PDF/TXT files → Click "🚀 Process Uploaded Files"
   ```
2. **Load Existing Vector Database Tab**
   - If you already processed documents before
   - Click "📥 Load Vector Database"
   - Loads from `./chroma_db` directory
   - Much faster than re-processing

**Use Case Example**:
```
Upload: company_policy.pdf, employee_handbook.pdf, FAQ.txt
Process: Creates searchable knowledge base
Result: Ready to answer questions about your documents
```

---
### Feature 2: 🔍 Query System Page

**Purpose**: Ask questions about your documents and send notifications

#### How It Works:

1. **Ask Your Question**
   ```
   Enter question: "What is the vacation policy?"
   ```

2. **Select Notification Channels**
   ```
   ☐ 📢 Send ALL (sends to all channels)
   ☐ 💬 WhatsApp
   ☐ 📱 SMS
   ☐ 📧 Email
   ☐ 🦅 Twitter
   ```

3. **Click "🚀 Submit Query"**

4. **Query Results**:
   ```
   📋 Query Results
   ├─ 💡 Summary (concise 2-3 sentences)
   ├─ 📄 Full Answer (expandable)
   ├─ 📚 Sources (which documents/pages were used)
   └─ 📤 Notification Status (success/failure for each channel)
   ```
   
   
### Feature 3: 📊 View History Page

**Purpose**: Track all your queries and processed documents

#### Two Tabs:

**Tab 1: 📝 Query Summaries**

1. Shows recent queries with timestamps
2. Adjustable limit (5-50 queries)
3. Each entry shows:
   - Question asked
   - Summary generated
   - Timestamp

**Tab 2: 📚 Document Metadata**

1. Shows all processed documents
2. Displays:
   - File name
   - File type (PDF/TXT badge)
   - Number of pages
   - Processing date

---

### Feature 4: ⚙️ Configuration Page

**Purpose**: View system settings and API configuration status

#### What You See:

1. **System Status**
   ```
   OpenAI Model: gpt-4
   Embedding Model: text-embedding-3-small
   Chunk Size: 1500
   Temperature: 0.7
   Retrieval K: 8
   Chunk Overlap: 300
   ```

2. **API Configuration Status**
   ```
   OpenAI API          ✅ Configured
   Twilio              ✅ Configured
   Email SMTP          ✅ Configured
   Twitter API         ❌ Not Configured
   ```

3. **Storage Locations**
   ```
   Vector Database: ./chroma_db
   SQLite Database: ./summaries.db
   Documents Directory: ./documents
   ```

4. **Notification Recipients**
   ```
   WhatsApp: +1234567890
   SMS: +1234567890
   Email: user@example.com
   ```

**Purpose**: Quick health check of your system configuration

---
