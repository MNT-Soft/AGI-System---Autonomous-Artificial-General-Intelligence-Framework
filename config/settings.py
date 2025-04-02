import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # Пути
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/agi_model.pt")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/system.log")
    
    # Параметры нейросети
    EMBEDDING_SIZE: int = int(os.getenv("EMBEDDING_SIZE", 768))
    HIDDEN_SIZE: int = int(os.getenv("HIDDEN_SIZE", 512))
    
    # API ключи
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Безопасность
    MAX_MEMORY_USAGE: float = float(os.getenv("MAX_MEMORY_USAGE", 0.8))
    ENABLE_CODE_EXEC: bool = os.getenv("ENABLE_CODE_EXEC", "True") == "True"

settings = Settings()