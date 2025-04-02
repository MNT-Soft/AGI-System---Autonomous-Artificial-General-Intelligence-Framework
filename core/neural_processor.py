import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from config import config
import logging

logger = logging.getLogger(__name__)

class NeuralProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        
        # Адаптивные слои
        self.attention = nn.MultiheadAttention(
            embed_dim=config.EMBED_SIZE,
            num_heads=config.N_HEADS
        )
        
        self.projection = nn.Sequential(
            nn.Linear(config.EMBED_SIZE, config.HIDDEN_SIZE),
            nn.GELU(),
            nn.LayerNorm(config.HIDDEN_SIZE)
        )
        
        # Эволюционные параметры
        self.complexity_threshold = 0.7
        self.current_layers = 1
        self.max_layers = 5
    
    def forward(self, inputs):
        """Обработка входных данных"""
        try:
            if isinstance(inputs, str):
                inputs = self.tokenizer(inputs, return_tensors="pt", 
                                     max_length=512, truncation=True)
            
            embeddings = self.encoder(**inputs).last_hidden_state
            attn_out, _ = self.attention(embeddings, embeddings, embeddings)
            output = self.projection(attn_out.mean(dim=1))
            
            # Очистка памяти
            torch.cuda.empty_cache()
            
            return output
            
        except Exception as e:
            logger.error(f"Ошибка обработки: {str(e)}")
            raise RuntimeError("Ошибка нейронной обработки") from e
    
    def evolve(self, current_complexity):
        """Адаптация архитектуры"""
        if (current_complexity > self.complexity_threshold and 
            self.current_layers < self.max_layers):
            
            try:
                new_layer = nn.Sequential(
                    nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE),
                    nn.GELU(),
                    nn.LayerNorm(config.HIDDEN_SIZE)
                )
                
                self.projection.add_module(f"layer_{self.current_layers}", new_layer)
                self.current_layers += 1
                logger.info("Архитектура расширена: добавлен слой %d", self.current_layers)
                return True
            except Exception as e:
                logger.error("Ошибка эволюции: %s", str(e))
                return False
        return False
    
    def save(self, path):
        """Безопасное сохранение модели"""
        try:
            temp_path = f"{path}.tmp"
            torch.save(self.state_dict(), temp_path)
            os.replace(temp_path, path)
            logger.info("Модель сохранена: %s", path)
        except Exception as e:
            logger.error("Ошибка сохранения модели: %s", str(e))
            raise
    
    def load(self, path):
        """Загрузка модели"""
        try:
            if os.path.exists(path):
                self.load_state_dict(torch.load(path))
                logger.info("Модель загружена: %s", path)
        except Exception as e:
            logger.error("Ошибка загрузки модели: %s", str(e))
            raise
