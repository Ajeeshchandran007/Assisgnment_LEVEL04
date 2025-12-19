"""
===========================================================
RAG Agent with Multi-Channel Notifications and PDF Support
===========================================================
A comprehensive RAG system with document processing (including PDFs),
vector storage, and automated notifications.
"""

import os
import sqlite3
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import sys
import time
# LangChain imports
# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    DirectoryLoader,
    PyPDFLoader
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Communication imports
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
import requests

# Configuration
try:
    from config import Config
except ImportError:
    print("❌ Error: config.py not found!")
    print("Please create config.py file with your API credentials.")
    sys.exit(1)


class RAGAgent:
    """
    RAG Agent with PDF support, vector storage, and multi-channel notifications
    
    Features:
    - Process PDF and TXT documents
    - Store embeddings in ChromaDB
    - Query using GPT-4
    - Save summaries to SQLite
    - Send notifications via WhatsApp, SMS, Email, and Twitter
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the RAG agent
        
        Args:
            config: Configuration object (uses default Config if None)
        """
        self.config = config or Config()
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = self.config.OPENAI_API_KEY
        
        # Initialize embeddings
        print("📄 Initializing embeddings model...")
        self.embeddings = OpenAIEmbeddings(model=self.config.EMBEDDING_MODEL)
        
        # Initialize LLM
        print("📄 Initializing language model...")
        self.llm = ChatOpenAI(
            model=self.config.OPENAI_MODEL,
            temperature=self.config.TEMPERATURE
        )
        
        # Vector store
        self.persist_directory = self.config.CHROMA_PERSIST_DIRECTORY
        self.vectorstore = None
        
        # SQLite setup
        self.db_path = self.config.SQLITE_DB_PATH
        self._init_sqlite()
        
        print("✅ RAG Agent initialized successfully!\n")
        
    def _init_sqlite(self):
        """Initialize SQLite database for storing summaries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create summaries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                summary TEXT NOT NULL,
                full_answer TEXT,
                sources TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create metadata table for tracking document processing
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_type TEXT,
                pages INTEGER,
                processed_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ SQLite database initialized at: {self.db_path}")
        
    def load_pdf_documents(self, pdf_paths: List[str]) -> List[Document]:
        """
        Load PDF documents from specified paths
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of Document objects
        """
        all_docs = []
        
        for pdf_path in pdf_paths:
            try:
                print(f"📄 Loading PDF: {pdf_path}")
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)
                
                # Store metadata
                self._store_document_metadata(pdf_path, "PDF", len(docs))
                
                print(f"   ✅ Loaded {len(docs)} pages from {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f"   ❌ Error loading {pdf_path}: {e}")
        
        return all_docs
    
    def load_text_documents(self, directory_path: str) -> List[Document]:
        """
        Load text documents from directory
        
        Args:
            directory_path: Path to directory containing text files
            
        Returns:
            List of Document objects
        """
        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            docs = loader.load()
            
            # Store metadata for each text file
            for doc in docs:
                source = doc.metadata.get('source', '')
                if source:
                    self._store_document_metadata(source, "TXT", 1)
            
            return docs
        except Exception as e:
            print(f"⚠️  Error loading text files: {e}")
            return []
    
    def load_documents_from_directory(self, directory_path: str = None) -> List[Document]:
        """
        Load all documents (TXT and PDF) from directory
        
        Args:
            directory_path: Path to documents directory (uses config default if None)
            
        Returns:
            List of all loaded Document objects
        """
        if directory_path is None:
            directory_path = self.config.DOCUMENTS_DIRECTORY
        
        print(f"\n{'='*70}")
        print(f" LOADING DOCUMENTS FROM: {directory_path} ".center(70))
        print(f"{'='*70}\n")
        
        # Create directory if it doesn't exist
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        
        all_documents = []
        
        # Load text files
        print("📁 Searching for text files...")
        txt_docs = self.load_text_documents(directory_path)
        if txt_docs:
            all_documents.extend(txt_docs)
            print(f"✅ Loaded {len(txt_docs)} text documents\n")
        else:
            print("ℹ️  No text files found\n")
        
        # Load PDF files
        print("📄 Searching for PDF files...")
        pdf_files = list(Path(directory_path).rglob("*.pdf"))
        if pdf_files:
            pdf_docs = self.load_pdf_documents([str(f) for f in pdf_files])
            all_documents.extend(pdf_docs)
            print(f"\n✅ Loaded {len(pdf_docs)} PDF pages from {len(pdf_files)} files\n")
        else:
            print("ℹ️  No PDF files found\n")
        
        return all_documents
    
    def _store_document_metadata(self, file_path: str, file_type: str, pages: int):
        """Store document processing metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO document_metadata (file_path, file_type, pages) VALUES (?, ?, ?)",
                (file_path, file_type, pages)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️  Could not store metadata: {e}")
    
    def process_and_store_documents(self, 
                                   documents: List[Document] = None, 
                                   directory_path: str = None,
                                   pdf_paths: List[str] = None):
        """
        Process documents and create embeddings in vector store
        
        Args:
            documents: Pre-loaded documents (optional)
            directory_path: Directory to load documents from (optional)
            pdf_paths: Specific PDF files to load (optional)
        """
        print(f"\n{'='*70}")
        print(" DOCUMENT PROCESSING ".center(70))
        print(f"{'='*70}\n")
        
        # Load documents based on provided parameters
        if documents is None:
            documents = []
            
            if pdf_paths:
                print("📂 Loading specified PDF files...\n")
                documents.extend(self.load_pdf_documents(pdf_paths))
            
            if directory_path or not pdf_paths:
                documents.extend(self.load_documents_from_directory(directory_path))
        
        if not documents:
            print("⚠️  No documents loaded!")
            print("💡 Please add PDF or TXT files to the documents directory.\n")
            return False
        
        print(f"{'='*70}")
        print(f" Total Documents Loaded: {len(documents)} ".center(70))
        print(f"{'='*70}\n")
        
        # Split documents into chunks
        print("✂️  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(splits)} text chunks")
        print(f"   (Chunk size: {self.config.CHUNK_SIZE}, Overlap: {self.config.CHUNK_OVERLAP})\n")
        
        # Create vector store with embeddings
        print("📊 Creating embeddings and storing in ChromaDB...")
        print("   (This may take a few minutes depending on document size...)\n")
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print(f"{'='*70}")
            print(" ✅ DOCUMENTS SUCCESSFULLY PROCESSED! ".center(70))
            print(f"{'='*70}")
            print(f"\n   📊 Vector database location: {self.persist_directory}")
            print(f"   📦 Total embeddings created: {len(splits)}\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error creating embeddings: {e}")
            return False
        
    def load_existing_vectorstore(self):
        """Load existing vector store from disk"""
        if os.path.exists(self.persist_directory):
            try:
                print("📊 Loading existing vector store...")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print("✅ Loaded existing vector store successfully!\n")
                return True
            except Exception as e:
                print(f"❌ Error loading vector store: {e}\n")
                return False
        else:
            print("⚠️  No existing vector store found.")
            print(f"   Expected location: {self.persist_directory}")
            print("💡 Please process documents first (Option 1 or 2).\n")
            return False
        
    def query(self, question: str, k: int = None) -> Dict[str, any]:
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            k: Number of relevant chunks to retrieve (uses config default if None)
            
        Returns:
            Dictionary with answer, summary, sources, and metadata
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Load or process documents first.")
        
        if k is None:
            k = self.config.RETRIEVAL_K
        
        print(f"\n{'='*70}")
        print(f" QUERYING RAG SYSTEM ".center(70))
        print(f"{'='*70}\n")
        print(f"🔍 Question: {question}\n")
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Create prompt template
        template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.
    Provide a comprehensive and detailed answer based on the context.

    Context:
    {context}

    Question: {question}

    Detailed Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Get answer
        print("🤖 Generating answer...")
        
        # Retrieve documents
        source_docs = retriever.invoke(question)
        
        # Format context
        context = format_docs(source_docs)
        
        # Create chain using LCEL
        chain = prompt | self.llm | StrOutputParser()
        
        # Invoke chain
        answer = chain.invoke({"context": context, "question": question})
        
        print("✅ Answer generated successfully!")
        
        # Create summary
        print("📝 Creating concise summary...")
        summary = self._create_summary(question, answer)
        
        # Prepare sources info
        sources_info = []
        for doc in source_docs:
            source_dict = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources_info.append(source_dict)
        
        # Store in SQLite
        self._store_summary(question, summary, answer, sources_info)
        
        print("✅ Summary created and stored!\n")
        
        return {
            "question": question,
            "answer": answer,
            "summary": summary,
            "sources": sources_info,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _create_summary(self, question: str, answer: str) -> str:
        """
        Create a concise summary of the Q&A
        
        Args:
            question: The original question
            answer: The full answer
            
        Returns:
            Concise 2-3 sentence summary
        """
        summary_prompt = f"""Create a clear and concise summary of the following Q&A in 2-3 sentences.
Focus on the key points and main takeaways.

Question: {question}

Answer: {answer}

Concise Summary (2-3 sentences):"""
        
        try:
            summary_response = self.llm.invoke(summary_prompt)
            return summary_response.content.strip()
        except Exception as e:
            print(f"⚠️  Error creating summary: {e}")
            # Fallback: Use first 300 characters of answer
            return answer[:300] + "..." if len(answer) > 300 else answer
    
    def _store_summary(self, query: str, summary: str, full_answer: str = None, sources: List[Dict] = None):
        """
        Store summary in SQLite database
        
        Args:
            query: The question asked
            summary: Concise summary
            full_answer: Complete answer
            sources: List of source information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            sources_str = str(sources) if sources else None
            
            cursor.execute(
                "INSERT INTO summaries (query, summary, full_answer, sources) VALUES (?, ?, ?, ?)",
                (query, summary, full_answer, sources_str)
            )
            conn.commit()
            conn.close()
            print(f"✅ Summary stored in SQLite database")
        except Exception as e:
            print(f"⚠️  Could not store summary: {e}")
    
    def get_summaries(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve recent summaries from database
        
        Args:
            limit: Number of summaries to retrieve
            
        Returns:
            List of summary dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, query, summary, timestamp FROM summaries ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "id": row[0],
                    "query": row[1],
                    "summary": row[2],
                    "timestamp": row[3]
                }
                for row in results
            ]
        except Exception as e:
            print(f"❌ Error retrieving summaries: {e}")
            return []
    
    def send_whatsapp(self, summary: str) -> bool:
        """
        Send summary via WhatsApp using Twilio
        
        Args:
            summary: The summary text to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.config.get_notification_config()["whatsapp"]
            client = Client(config["twilio_account_sid"], config["twilio_auth_token"])
            
            message_body = f"📊 RAG Agent Summary\n\n{summary}\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            message = client.messages.create(
                body=message_body,
                from_=f"whatsapp:{config['twilio_whatsapp_number']}",
                to=f"whatsapp:{config['recipient_whatsapp']}"
            )
            
            print(f"✅ WhatsApp sent successfully!")
            print(f"   Message SID: {message.sid}")
            return True
        except Exception as e:
            print(f"❌ WhatsApp error: {e}")
            return False
    
    def send_sms(self, summary: str) -> bool:
        """
        Send summary via SMS using Twilio with delivery tracking
        
        Args:
            summary: The summary text to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.config.get_notification_config()["sms"]
            client = Client(config["twilio_account_sid"], config["twilio_auth_token"])
            
            # SMS has character limits, so truncate if needed
            #sms_text = f"RAG Summary: {summary}"
            sms_text = f"RAG: {summary[:140]}"  
            if len(sms_text) > 150:
                sms_text = sms_text[:150] + "..."
            
            message = client.messages.create(
                body=sms_text,
                from_=config["twilio_phone_number"],
                to=config["recipient_phone"]
            )
            
            print(f"✅ SMS sent successfully!")
            print(f"   Message SID: {message.sid}")
            print(f"   Status: {message.status}")
            print(f"   To: {config['recipient_phone']}")
            print(f"   From: {config['twilio_phone_number']}")
            
            # Check message status after a brief delay
            import time
            time.sleep(3)
            
            # Fetch updated message status
            message_status = client.messages(message.sid).fetch()
            print(f"   Updated Status: {message_status.status}")
            
            if message_status.error_code:
                print(f"   ⚠️ Error Code: {message_status.error_code}")
                print(f"   Error Message: {message_status.error_message}")
                return False
            
            # Check for Indian phone number specific issues
            if config["recipient_phone"].startswith("+91"):
                print(f"\n   ℹ️ Indian Number Detected:")
                print(f"   - Make sure DND (Do Not Disturb) is disabled")
                print(f"   - Check if international SMS is enabled")
                print(f"   - Try registering your sender ID with TRAI")
            
            return True
        except Exception as e:
            print(f"❌ SMS error: {e}")
            print(f"\n💡 Troubleshooting:")
            print(f"   1. Check Twilio logs: https://console.twilio.com/us1/monitor/logs/sms")
            print(f"   2. Verify phone number format: {config.get('recipient_phone', 'Not set')}")
            print(f"   3. Check if number is on DND list (India)")
            print(f"   4. Try using Twilio's messaging service instead")
            return False
    
    def send_email(self, summary: str, full_answer: str = None) -> bool:
        """
        Send summary via Email (Exchange compatible)
        
        Args:
            summary: The summary text
            full_answer: Optional full answer to include
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.config.get_notification_config()["email"]
            
            msg = MIMEMultipart("alternative")
            msg["From"] = config["email_sender"]
            msg["To"] = config["email_recipient"]
            msg["Subject"] = f"RAG Agent Summary - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Create HTML email
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 600px; margin: 0 auto; background-color: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }}
                    .header h1 {{ margin: 0; font-size: 24px; }}
                    .content {{ padding: 30px; }}
                    .summary-box {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; margin: 20px 0; }}
                    .answer-box {{ background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                    .footer {{ text-align: center; padding: 20px; color: #6c757d; font-size: 12px; border-top: 1px solid #e9ecef; }}
                    .timestamp {{ color: #6c757d; font-size: 14px; margin-top: 15px; }}
                    h2 {{ color: #333; font-size: 18px; margin-top: 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 RAG Agent Summary</h1>
                    </div>
                    <div class="content">
                        <h2>Summary</h2>
                        <div class="summary-box">
                            <p style="margin: 0; line-height: 1.6; font-size: 16px;">{summary}</p>
                        </div>
                        
                        {f'''
                        <h2>Full Answer</h2>
                        <div class="answer-box">
                            <p style="margin: 0; line-height: 1.6;">{full_answer}</p>
                        </div>
                        ''' if full_answer else ''}
                        
                        <div class="timestamp">
                            ⏰ Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        </div>
                    </div>
                    <div class="footer">
                        <p>This email was sent by your RAG Agent system</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, "html"))
            
            # Try different connection methods for Exchange
            smtp_port = config["smtp_port"]
            
            try:
                # Method 1: Try STARTTLS (port 587)
                if smtp_port == 587:
                    print(f"   Connecting to {config['smtp_server']}:{smtp_port} (STARTTLS)...")
                    with smtplib.SMTP(config["smtp_server"], smtp_port, timeout=10) as server:
                        server.starttls()
                        server.login(config["email_sender"], config["email_password"])
                        server.send_message(msg)
                
                # Method 2: Try SSL (port 465)
                elif smtp_port == 465:
                    print(f"   Connecting to {config['smtp_server']}:{smtp_port} (SSL)...")
                    with smtplib.SMTP_SSL(config["smtp_server"], smtp_port, timeout=10) as server:
                        server.login(config["email_sender"], config["email_password"])
                        server.send_message(msg)
                
                # Method 3: Try basic SMTP (port 25) - no encryption
                else:
                    print(f"   Connecting to {config['smtp_server']}:{smtp_port} (Basic SMTP)...")
                    with smtplib.SMTP(config["smtp_server"], smtp_port, timeout=10) as server:
                        # Try to login if credentials provided
                        try:
                            server.login(config["email_sender"], config["email_password"])
                        except smtplib.SMTPException:
                            # Some Exchange servers don't require auth on port 25
                            print("   Note: Auth not required or failed, sending anyway...")
                        server.send_message(msg)
                
                print(f"✅ Email sent successfully!")
                print(f"   From: {config['email_sender']}")
                print(f"   To: {config['email_recipient']}")
                return True
                
            except Exception as conn_error:
                print(f"   ❌ Connection failed: {conn_error}")
                
                # If port 587 or 465 failed, suggest trying port 25
                if smtp_port in [587, 465]:
                    print(f"   💡 Try changing SMTP_PORT to 25 in config.py")
                
                raise conn_error
                
        except Exception as e:
            print(f"❌ Email error: {e}")
            print(f"\n💡 Troubleshooting:")
            print(f"   1. Verify SMTP server: exch01.domain.com is reachable")
            print(f"   2. Try different ports: 25, 587, or 465")
            print(f"   3. Check if firewall is blocking SMTP")
            print(f"   4. Verify credentials: {config['email_sender']}")
            return False
    
    def post_to_twitter(self, summary: str) -> bool:
        """
        Post summary to Twitter (X) using OAuth 1.0a
        
        Args:
            summary: The summary text to post
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from requests_oauthlib import OAuth1
            
            # Shorten summary to 280 characters
            short_summary = self._shorten_to_280(summary)
            
            config = self.config.get_notification_config()["twitter"]
            
            # Twitter API v2 endpoint with OAuth 1.0a
            url = "https://api.twitter.com/2/tweets"
            
            # Create OAuth1 authentication
            auth = OAuth1(
                config['twitter_api_key'],
                config['twitter_api_secret'],
                config['twitter_access_token'],
                config['twitter_access_secret']
            )
            
            data = {
                "text": short_summary
            }
            
            response = requests.post(url, auth=auth, json=data)
            
            if response.status_code == 201:
                tweet_data = response.json()
                tweet_id = tweet_data.get("data", {}).get("id", "Unknown")
                print(f"✅ Tweet posted successfully!")
                print(f"   Tweet ID: {tweet_id}")
                print(f"   Content: {short_summary[:50]}...")
                return True
            else:
                print(f"❌ Twitter error: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Twitter error: {e}")
            return False
    
    def _shorten_to_280(self, text: str) -> str:
        """
        Shorten text to 280 characters for Twitter
        
        Args:
            text: The text to shorten
            
        Returns:
            Text shortened to 280 characters or less
        """
        if len(text) <= 280:
            return text
        
        # Use LLM to create a concise 280-char summary
        shorten_prompt = f"""Shorten this text to EXACTLY 280 characters or less while preserving the key message.
Be concise, clear, and impactful. Do not use quotes or extra formatting.

Text: {text}

Shortened version (max 280 chars):"""
        
        try:
            response = self.llm.invoke(shorten_prompt)
            shortened = response.content.strip()
            
            # Remove quotes if present
            shortened = shortened.strip('"').strip("'")
            
            # Ensure it's under 280
            if len(shortened) > 280:
                shortened = shortened[:277] + "..."
            
            return shortened
        except Exception as e:
            print(f"⚠️  Error shortening text: {e}")
            # Fallback: Simple truncation
            return text[:277] + "..." if len(text) > 280 else text
    
    def process_query_with_notifications(self, 
                                        question: str, 
                                        send_whatsapp: bool = True,
                                        send_sms: bool = True,
                                        send_email: bool = True,
                                        send_twitter: bool = True) -> Dict:
        """
        Query the RAG system and send notifications to selected channels
        
        Args:
            question: The question to ask
            send_whatsapp: Whether to send WhatsApp notification
            send_sms: Whether to send SMS notification
            send_email: Whether to send email notification
            send_twitter: Whether to post to Twitter
            
        Returns:
            Dictionary with query results and notification status
        """
        print(f"\n{'='*70}")
        print(" PROCESSING QUERY WITH NOTIFICATIONS ".center(70))
        print(f"{'='*70}\n")
        
        # Get answer from RAG
        result = self.query(question)
        summary = result["summary"]
        answer = result["answer"]
        
        print(f"\n{'='*70}")
        print(" SUMMARY ".center(70))
        print(f"{'='*70}")
        print(f"\n{summary}\n")
        print(f"{'='*70}\n")
        
        # Send notifications
        print("📤 Sending Notifications...\n")
        
        notification_results = {}
        
        if send_whatsapp:
            print("📱 Sending to WhatsApp...")
            notification_results['whatsapp'] = self.send_whatsapp(summary)
            print()
        
        if send_sms:
            print("💬 Sending SMS...")
            notification_results['sms'] = self.send_sms(summary)
            print()
        
        if send_email:
            print("📧 Sending Email...")
            notification_results['email'] = self.send_email(summary, answer)
            print()
        
        if send_twitter:
            print("🦅 Posting to Twitter...")
            notification_results['twitter'] = self.post_to_twitter(summary)
            print()
        
        print(f"{'='*70}")
        print(" ✅ PROCESSING COMPLETE! ".center(70))
        print(f"{'='*70}\n")
        
        # Add notification results to return value
        result['notifications'] = notification_results
        
        return result


# Example usage with better logging
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" RAG AGENT - DIRECT USAGE DEMO ".center(70))
    print("="*70 + "\n")
    
    print("🔧 Step 1: Validating Configuration...")
    print("-" * 70)
    Config.validate_config()
    
    print("\n🔧 Step 2: Initializing RAG Agent...")
    print("-" * 70)
    try:
        agent = RAGAgent(Config)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    
    print("\n🔧 Step 3: Checking for existing vector store...")
    print("-" * 70)
    
    # Try to load existing vector store
    if not agent.load_existing_vectorstore():
        print("\n🔧 Step 4: Processing documents from directory...")
        print("-" * 70)
        success = agent.process_and_store_documents()
        
        if not success:
            print("\n⚠️  No documents to process!")
            print(f"💡 Add PDF or TXT files to: {Config.DOCUMENTS_DIRECTORY}")
            print("💡 Then run this script again.")
            sys.exit(0)
    
    # Example query if vector store is loaded
    if agent.vectorstore:
        print("\n🔧 Step 5: Running example query...")
        print("-" * 70)
        
        test_question = "What are the main topics discussed in the documents?"
        print(f"\n❓ Test Question: {test_question}\n")
        
        result = agent.process_query_with_notifications(
            question=test_question,
            send_whatsapp=False,  # Set to True when configured
            send_sms=True,       # Set to True when configured
            send_email=False,     # Set to True when configured
            send_twitter=True    # Set to True when configured
        )
        
        print("\n" + "="*70)
        print(" 📋 QUERY RESULTS ".center(70))
        print("="*70)
        print(f"\n🔍 Question: {result['question']}")
        print(f"\n💡 Summary: {result['summary']}")
        print(f"\n📚 Full Answer:")
        print("-" * 70)
        print(result['answer'])
        print("-" * 70)
        print(f"\n📖 Sources: {len(result['sources'])} documents")
        print(f"⏰ Timestamp: {result['timestamp']}")