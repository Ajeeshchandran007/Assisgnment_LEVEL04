"""
Configuration Management for RAG Agent
======================================
Minimal configuration for local RAG system with Ollama and ChromaDB.
"""

import os


class Config:
    """Central configuration for RAG Agent"""
    
    # ==================== OLLAMA CONFIGURATION ====================
    # Local LLM settings
    OLLAMA_MODEL = "llama3.2"
    EMBEDDING_MODEL = "nomic-embed-text"
    TEMPERATURE = 0.7
    NUM_CTX = 4096  # Context window size
    
    # ==================== VECTOR DATABASE CONFIGURATION ====================
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    
    # ==================== DOCUMENT PROCESSING CONFIGURATION ====================
    DOCUMENTS_DIRECTORY = "./documents"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    RETRIEVAL_K = 5  # Number of relevant chunks to retrieve
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configurations are set"""
        print("\n" + "="*70)
        print(" CONFIGURATION VALIDATION ".center(70))
        print("="*70 + "\n")
        
        print("✅ Configuration validated successfully!")
        print(f"   - Ollama Model: {cls.OLLAMA_MODEL}")
        print(f"   - Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"   - Vector DB: {cls.CHROMA_PERSIST_DIRECTORY}")
        print(f"   - Documents Dir: {cls.DOCUMENTS_DIRECTORY}")
        
        print("\n💡 Make sure you have pulled the required Ollama models:")
        print(f"   ollama pull {cls.OLLAMA_MODEL}")
        print(f"   ollama pull {cls.EMBEDDING_MODEL}")
        
        print("\n" + "="*70 + "\n")
        return True
    
    @classmethod
    def print_config_status(cls):
        """Print current configuration status"""
        print("\n" + "="*70)
        print(" CURRENT CONFIGURATION ".center(70))
        print("="*70 + "\n")
        
        configs = [
            ("Ollama Model", cls.OLLAMA_MODEL),
            ("Embedding Model", cls.EMBEDDING_MODEL),
            ("Temperature", cls.TEMPERATURE),
            ("Context Window", cls.NUM_CTX),
            ("Vector DB Location", cls.CHROMA_PERSIST_DIRECTORY),
            ("Documents Directory", cls.DOCUMENTS_DIRECTORY),
            ("Chunk Size", cls.CHUNK_SIZE),
            ("Chunk Overlap", cls.CHUNK_OVERLAP),
            ("Retrieval K", cls.RETRIEVAL_K),
        ]
        
        for name, value in configs:
            print(f"   {name:.<50} {value}")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Test configuration
    Config.print_config_status()
    Config.validate_config()