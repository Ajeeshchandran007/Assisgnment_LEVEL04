"""
Configuration Management for RAG Agent
======================================
Store all API keys and configuration in this file.
"""

import os
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Config:
    """Central configuration for RAG Agent"""
    
    # ==================== OPENAI CONFIGURATION ====================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4"  
    EMBEDDING_MODEL = "text-embedding-3-small"
    TEMPERATURE = 0.7
    
    # ==================== VECTOR DATABASE CONFIGURATION ====================
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    
    # ==================== SQLITE CONFIGURATION ====================
    SQLITE_DB_PATH = "./summaries.db"
    
    # ==================== DOCUMENT PROCESSING CONFIGURATION ====================
    DOCUMENTS_DIRECTORY = "./documents"
    CHUNK_SIZE = 1500  
    CHUNK_OVERLAP = 300
    RETRIEVAL_K = 8  # Number of relevant chunks to retrieve
    
    # ==================== TWILIO CONFIGURATION ====================

    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "AC4b5b03d64478a5c9aa3444efd7183a9d")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "c24ec93fecc129361239ac6dc227ced1")
    
    # For WhatsApp: Use Twilio WhatsApp Sandbox number
    TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "+14155238886")
    
    # For SMS: Your Twilio phone number
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+14056484984")
    
    # ==================== RECIPIENT CONFIGURATION ====================
    # Your phone number (with country code, e.g., +1234567890)
    RECIPIENT_WHATSAPP = os.getenv("RECIPIENT_WHATSAPP", "+919900720451")
    RECIPIENT_PHONE = os.getenv("RECIPIENT_PHONE", "+919900720451")
    
    # ==================== EMAIL CONFIGURATION ====================
    # For Gmail: Enable 2FA and create App Password
    # Guide: https://support.google.com/accounts/answer/185833
    SMTP_SERVER = os.getenv("SMTP_SERVER", "exch01.domain.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    EMAIL_SENDER = os.getenv("EMAIL_SENDER", "alanaegaymon@domain.com")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "qampass1!")
    EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "es1service@domain.com")
    
    # ==================== TWITTER CONFIGURATION ====================
    # Get credentials from: https://developer.twitter.com/en/portal/dashboard
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "AAAAAAAAAAAAAAAAAAAAAAQ95wEAAAAAs218r0PyyucxriMO9vSUbt3dUBM%3DKX0g3ivEIR7L246SMa1vrAu1JdOB0JmyMsi3SaYq3hHkIUGoSQ")
    TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "23BQIx4Vtyi1wYrqQdXSrYbm8")
    TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "T0jax9UYD9XTNB9b02dVWpWYj7T1vrjnRjys1UkBQjTbYLmlju")
    TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "1996453749708193792-6BuvrCBadJsm0W8Z70kzcsRg2cY8va")
    TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET", "kHrNsbxcGQ9SubPlrw7tOmZhmjSUfMi9l5LyY7sHfyoTA")
    
    @classmethod
    def get_notification_config(cls) -> Dict:
        """Get all notification configurations as a dictionary"""
        return {
            "whatsapp": {
                "twilio_account_sid": cls.TWILIO_ACCOUNT_SID,
                "twilio_auth_token": cls.TWILIO_AUTH_TOKEN,
                "twilio_whatsapp_number": cls.TWILIO_WHATSAPP_NUMBER,
                "recipient_whatsapp": cls.RECIPIENT_WHATSAPP
            },
            "sms": {
                "twilio_account_sid": cls.TWILIO_ACCOUNT_SID,
                "twilio_auth_token": cls.TWILIO_AUTH_TOKEN,
                "twilio_phone_number": cls.TWILIO_PHONE_NUMBER,
                "recipient_phone": cls.RECIPIENT_PHONE
            },
            "email": {
                "smtp_server": cls.SMTP_SERVER,
                "smtp_port": cls.SMTP_PORT,
                "email_sender": cls.EMAIL_SENDER,
                "email_password": cls.EMAIL_PASSWORD,
                "email_recipient": cls.EMAIL_RECIPIENT
            },
            "twitter": {
                "twitter_bearer_token": cls.TWITTER_BEARER_TOKEN,
                "twitter_api_key": cls.TWITTER_API_KEY,
                "twitter_api_secret": cls.TWITTER_API_SECRET,
                "twitter_access_token": cls.TWITTER_ACCESS_TOKEN,
                "twitter_access_secret": cls.TWITTER_ACCESS_SECRET
            }
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configurations are set"""
        print("\n" + "="*70)
        print(" CONFIGURATION VALIDATION ".center(70))
        print("="*70 + "\n")
        
        required_configs = [
            ("OpenAI API Key", cls.OPENAI_API_KEY, "https://platform.openai.com/api-keys"),
            ("Twilio Account SID", cls.TWILIO_ACCOUNT_SID, "https://console.twilio.com"),
            ("Twilio Auth Token", cls.TWILIO_AUTH_TOKEN, "https://console.twilio.com"),
            ("Email Sender", cls.EMAIL_SENDER, "alanaegaymon@domain.com"),
            ("Email Password", cls.EMAIL_PASSWORD, "qampass1!"),
        ]
        
        missing = []
        for name, value, hint in required_configs:
            if value.startswith("your-") or not value or len(value) < 8:
                missing.append((name, hint))
        
        if missing:
            print("⚠️  MISSING OR INVALID CONFIGURATIONS:\n")
            for name, hint in missing:
                print(f"   ❌ {name}")
                print(f"      Get it from: {hint}\n")
            
            print("="*70)
            print("\n💡 Update config.py with your actual credentials before proceeding.\n")
            return False
        
        print("✅ All required configurations are set!")
        print("="*70 + "\n")
        return True
    
    @classmethod
    def print_config_status(cls):
        """Print current configuration status"""
        print("\n" + "="*70)
        print(" CURRENT CONFIGURATION ".center(70))
        print("="*70 + "\n")
        
        configs = [
            ("OpenAI Model", cls.OPENAI_MODEL),
            ("Embedding Model", cls.EMBEDDING_MODEL),
            ("Vector DB Location", cls.CHROMA_PERSIST_DIRECTORY),
            ("SQLite DB Location", cls.SQLITE_DB_PATH),
            ("Documents Directory", cls.DOCUMENTS_DIRECTORY),
            ("Chunk Size", cls.CHUNK_SIZE),
            ("Chunk Overlap", cls.CHUNK_OVERLAP),
            ("Recipient WhatsApp", cls.RECIPIENT_WHATSAPP),
            ("Recipient Phone", cls.RECIPIENT_PHONE),
            ("Email Recipient", cls.EMAIL_RECIPIENT),
        ]
        
        for name, value in configs:
            print(f"   {name:.<50} {value}")
        
        print("\n" + "="*70 + "\n")



if __name__ == "__main__":
    # Test configuration
    Config.print_config_status()
    Config.validate_config()