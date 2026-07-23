import logging
import os
from pathlib import Path
from typing import List
from pydantic import model_validator, ConfigDict
from pydantic_settings import BaseSettings

# Define Settings Class
class Settings(BaseSettings):
    # LLM Settings
    LLM_PROVIDER: str = ""        # Set dynamically in validator if left empty
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = ""           # Automatically mapped based on provider if left empty
    
    # Embedding & Reranker Models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = ""      # Disabled by default for low memory (set to BAAI/bge-reranker-base if needed)
    
    # Ollama Specifics
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"
    
    # Retrieval Hyperparameters
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    RETRIEVAL_TOP_K: int = 15
    RERANK_TOP_K: int = 4
    HYBRID_ALPHA: float = 0.5
    VERIFICATION_THRESHOLD: float = 0.6
    
    # Security and API Configuration
    ALLOWED_ORIGINS: str = "*"     # Comma-separated or "*"
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_UPLOAD_SIZE: int = 20971520  # 20MB in bytes
    FRONTEND_TIMEOUT: float = 60.0   # Request timeout for client
    
    # Persistence Directories
    DATA_DIR: Path = Path("./data")
    CHROMA_DB_DIR: Path = Path("./data/chroma_db")
    UPLOAD_DIR: Path = Path("./data/uploads")
    SQLITE_DB_PATH: Path = Path("./data/rag_system.db")
    HF_HOME: Path = Path("./data/hf_cache")
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    @property
    def allowed_origins_list(self) -> List[str]:
        if not self.ALLOWED_ORIGINS or self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @model_validator(mode="after")
    def validate_and_setup(self) -> "Settings":
        # Resolve absolute paths
        self.DATA_DIR = self.DATA_DIR.resolve()
        self.CHROMA_DB_DIR = self.CHROMA_DB_DIR.resolve()
        self.UPLOAD_DIR = self.UPLOAD_DIR.resolve()
        self.SQLITE_DB_PATH = self.SQLITE_DB_PATH.resolve()
        self.HF_HOME = self.HF_HOME.resolve()
        
        # Create directories
        for directory in [self.DATA_DIR, self.CHROMA_DB_DIR, self.UPLOAD_DIR, self.HF_HOME]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Clean LLM Provider
        if not self.LLM_PROVIDER.strip():
            self.LLM_PROVIDER = "groq" if self.GROQ_API_KEY.strip() else "ollama"
        self.LLM_PROVIDER = self.LLM_PROVIDER.lower().strip()
        if self.LLM_PROVIDER not in ["ollama", "groq"]:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.LLM_PROVIDER}. Must be 'ollama' or 'groq'.")
            
        # Validate keys based on provider
        if self.LLM_PROVIDER == "groq" and not self.GROQ_API_KEY.strip():
            raise ValueError("GROQ_API_KEY must be set when LLM_PROVIDER is 'groq'.")
            
        # Map default model if empty
        if not self.LLM_MODEL.strip():
            if self.LLM_PROVIDER == "groq":
                self.LLM_MODEL = "mixtral-8x7b-32768"
            else:
                self.LLM_MODEL = self.OLLAMA_MODEL
                
        return self

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate Singleton settings object loaded at startup
settings = Settings()
print(settings.model_dump())

# Set HuggingFace Cache Directory before loading transformers/sentence-transformers
os.environ["HF_HOME"] = str(settings.HF_HOME)

# Initialize standard logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger("rag_system.config")
logger.info(f"Configuration loaded. Active LLM Provider: {settings.LLM_PROVIDER.upper()}, Model: {settings.LLM_MODEL}")
