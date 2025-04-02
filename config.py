import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Пути
    MODEL_PATH: str = "models/agi_model.pth"
    LOG_FILE: str = "logs/system.log"
    KNOWLEDGE_GRAPH: str = "data/knowledge_graph.gml"
    
    # Параметры нейросети
    EMBEDDING_SIZE: int = 768
    HIDDEN_SIZE: int = 512
    NUM_ATTENTION_HEADS: int = 4
    
    # Обучение
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    MAX_ITERATIONS: int = 1000
    
    # Безопасность
    MAX_MEMORY_USAGE: float = 0.8  # 80% от доступной памяти
    MAX_CODE_EXECUTION_TIME: int = 5  # секунд
    
    @classmethod
    def from_env(cls):
        """Загрузка конфигурации из переменных окружения"""
        return cls(
            MODEL_PATH=os.getenv("MODEL_PATH", cls.MODEL_PATH)
        )