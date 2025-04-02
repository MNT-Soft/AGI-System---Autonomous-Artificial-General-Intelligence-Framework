import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AGIConfig:
    # Пути
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/agi_model.pt")
    LOG_PATH: str = os.getenv("LOG_PATH", "logs/system.log")
    KNOWLEDGE_PATH: str = os.getenv("KNOWLEDGE_PATH", "data/knowledge.gml")
    
    # Параметры нейросети
    EMBED_SIZE: int = int(os.getenv("EMBED_SIZE", 768))
    HIDDEN_SIZE: int = int(os.getenv("HIDDEN_SIZE", 512))
    N_HEADS: int = int(os.getenv("N_HEADS", 4))
    
    # Обучение
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 32))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", 0.001))
    
    # Безопасность
    MAX_MEMORY: float = float(os.getenv("MAX_MEMORY", 0.8))
    MAX_CODE_EXEC_TIME: int = int(os.getenv("MAX_CODE_EXEC_TIME", 10))
    
    # API
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

config = AGIConfig()