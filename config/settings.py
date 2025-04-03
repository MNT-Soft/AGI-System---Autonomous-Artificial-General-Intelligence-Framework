import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    """Configuration for the AGI system."""
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/organism.pt")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/organism.log")
    KNOWLEDGE_GRAPH_PATH: str = os.getenv("KNOWLEDGE_GRAPH_PATH", "knowledge_graph.gml")
    DATA_DB_PATH: str = os.getenv("DATA_DB_PATH", "data.db")
    
    EMBEDDING_SIZE: int = int(os.getenv("EMBEDDING_SIZE", 768))
    HIDDEN_SIZE: int = int(os.getenv("HIDDEN_SIZE", 512))
    META_LR: float = float(os.getenv("META_LR", 0.01))
    
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", None)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", None)
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", None)
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", None)
    
    MAX_MEMORY_USAGE: float = float(os.getenv("MAX_MEMORY_USAGE", 0.95))
    ENABLE_CODE_EXEC: bool = os.getenv("ENABLE_CODE_EXEC", "True") == "True"
    DEBUG: bool = os.getenv("DEBUG", "False") == "True"
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 32))
    GENETIC_POPULATION: int = int(os.getenv("GENETIC_POPULATION", 10))
    SOCIAL_PORT: int = int(os.getenv("SOCIAL_PORT", 5555))

    def __post_init__(self):
        if self.EMBEDDING_SIZE <= 0 or self.HIDDEN_SIZE <= 0:
            raise ValueError("EMBEDDING_SIZE and HIDDEN_SIZE must be positive")
        if self.BATCH_SIZE <= 0 or self.GENETIC_POPULATION <= 0:
            raise ValueError("BATCH_SIZE and GENETIC_POPULATION must be positive")
        if self.META_LR <= 0:
            raise ValueError("META_LR must be positive")

settings = Settings()