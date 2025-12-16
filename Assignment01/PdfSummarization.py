"""
PDF Summarization using Llama 3.2, LangChain, ChromaDB and RAG
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os


class PDFSummarizer:
    def __init__(self, model_name="llama3.2"):
        """Initialize the PDF Summarizer with Llama 3.2"""
        print("Initializing PDF Summarizer...")
        
        # Initialize Llama 3.2 LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3  # Lower temperature for more focused summaries
        )
        
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=model_name)
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.vectorstore = None
        self.rag_chain = None
        
    def load_and_process_pdf(self, pdf_path):
        """Load PDF and create vector store"""
        print(f"\nLoading PDF: {pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks")
        
        # Create vector store with ChromaDB
        print("Creating ChromaDB vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            collection_name="pdf_collection"
        )
        
        # Create RAG chain
        self._create_rag_chain()
        print("Vector store created successfully!")
        
    def _create_rag_chain(self):
        """Create the RAG chain using LCEL (LangChain Expression Language)"""
        # Define prompt template
        template = """Use the following pieces of context to answer the question. 
        If you don't know the answer, just say that you don't know, don't make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer: """
        
        prompt = PromptTemplate.from_template(template)
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create RAG chain using LCEL
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def summarize(self, summary_type="comprehensive"):
        """Generate summary of the PDF document"""
        print(f"\nGenerating {summary_type} summary...")
        
        summary_prompts = {
            "comprehensive": "Provide a comprehensive summary of the entire document, covering all main topics and key points.",
            "brief": "Provide a brief 2-3 paragraph summary highlighting the most important points of the document.",
            "bullet_points": "Summarize the document in 7-10 bullet points covering the main ideas.",
            "abstract": "Write an abstract-style summary of the document as if for an academic paper."
        }
        
        query = summary_prompts.get(summary_type, summary_prompts["comprehensive"])
        result = self.rag_chain.invoke(query)
        
        return result
    
    def ask_question(self, question):
        """Ask a specific question about the PDF content"""
        print(f"\nAnswering question: {question}")
        result = self.rag_chain.invoke(question)
        return result
    
    def get_key_topics(self):
        """Extract key topics from the document"""
        query = "What are the main topics and themes discussed in this document? List them clearly."
        result = self.rag_chain.invoke(query)
        return result
    
    def cleanup(self):
        """Clean up vector store"""
        if self.vectorstore:
            try:
                self.vectorstore.delete_collection()
            except:
                pass


def main():
    # Load all PDFs from the documents folder
    documents_folder = "./documents"
    
    # Check if documents folder exists
    if not os.path.exists(documents_folder):
        print(f"Error: '{documents_folder}' folder not found!")
        print("Please create a 'documents' folder and add your PDF files there.")
        return
    
    # Get all PDF files from the documents folder
    pdf_files = [f for f in os.listdir(documents_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{documents_folder}' folder!")
        return
    
    print(f"\nFound {len(pdf_files)} PDF file(s):")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"{i}. {pdf}")
    
    # Let user select which PDF to summarize
    if len(pdf_files) == 1:
        selected_pdf = pdf_files[0]
        print(f"\nAutomatically selecting: {selected_pdf}")
    else:
        print("\nEnter the number of the PDF you want to summarize (or press Enter for the first one): ", end='')
        choice = input().strip()
        
        if choice == '':
            selected_pdf = pdf_files[0]
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(pdf_files):
                    selected_pdf = pdf_files[idx]
                else:
                    print("Invalid selection. Using first PDF.")
                    selected_pdf = pdf_files[0]
            except ValueError:
                print("Invalid input. Using first PDF.")
                selected_pdf = pdf_files[0]
    
    pdf_path = os.path.join(documents_folder, selected_pdf)
    
    # Initialize summarizer
    summarizer = PDFSummarizer(model_name="llama3.2")
    
    # Load and process PDF
    try:
        summarizer.load_and_process_pdf(pdf_path)
        
        # Generate different types of summaries
        print("\n" + "="*80)
        print("COMPREHENSIVE SUMMARY")
        print("="*80)
        comprehensive = summarizer.summarize("comprehensive")
        print(comprehensive)
        
        print("\n" + "="*80)
        print("BRIEF SUMMARY")
        print("="*80)
        brief = summarizer.summarize("brief")
        print(brief)
        
        print("\n" + "="*80)
        print("KEY TOPICS")
        print("="*80)
        topics = summarizer.get_key_topics()
        print(topics)
        
        # Ask custom questions
        print("\n" + "="*80)
        print("CUSTOM QUESTION")
        print("="*80)
        answer = summarizer.ask_question("What are the main conclusions?")
        print(answer)
        
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        print("Please update the pdf_path variable with your PDF file location.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        summarizer.cleanup()


if __name__ == "__main__":
    main()