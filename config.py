# ============================================================================
# FILE: config.py
# ============================================================================
"""Configuration settings for the AI Agent"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""
    
    # LLM Settings
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TEMPERATURE = 0.7
    OLLAMA_TOP_P = 0.9
    
    # Vector Database
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    COLLECTION_NAME = "conversation_transcripts"
    TOP_K_RESULTS = 3
    
    # Voice Processing
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
    RECORDING_DURATION = 5  # seconds
    
    # Application Settings
    APP_PORT = int(os.getenv("APP_PORT", "7860"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
    
    # Google Gemini (Optional)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = "gemini-pro"
    
    # Paths
    DATA_DIR = "./data"
    LOGS_DIR = "./logs"
    UPLOADS_DIR = os.path.join(DATA_DIR, "user_uploads")
    
    # Intent Categories
    INTENT_CATEGORIES = [
        "customer_query",
        "complaint",
        "feedback",
        "support_request",
        "product_inquiry",
        "billing_issue",
        "general_conversation"
    ]
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        os.makedirs(cls.UPLOADS_DIR, exist_ok=True)
        os.makedirs(cls.CHROMA_PERSIST_DIR, exist_ok=True)


