import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class AdaptiveNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Текстовый процессор
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.attention = nn.MultiheadAttention(
            embed_dim=config.EMBEDDING_SIZE,
            num_heads=config.NUM_ATTENTION_HEADS
        )
        
        # Эволюционные параметры
        self.current_capacity = 1
        self.max_capacity = 10
        
        # Динамические слои
        self.layers = nn.ModuleList([
            nn.Linear(config.EMBEDDING_SIZE, config.HIDDEN_SIZE),
            nn.ReLU()
        ])
        
    def forward(self, x):
        # Токенизация текста
        if isinstance(x, str):
            x = self._tokenize(x)
            
        # Обработка изображений
        elif isinstance(x, torch.Tensor) and len(x.shape) == 4:
            x = self._process_image(x)
            
        # Прохождение через сеть
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def evolve(self, complexity: float):
        """Адаптация архитектуры под сложность задачи"""
        if complexity > 0.7 and self.current_capacity < self.max_capacity:
            new_size = int(self.config.HIDDEN_SIZE * 1.2)
            self.layers.append(nn.Linear(self.config.HIDDEN_SIZE, new_size))
            self.layers.append(nn.ReLU())
            self.current_capacity += 1
            return True
        return False
    
    def _tokenize(self, text: str):
        """Токенизация текста через BERT"""
        inputs = self.tokenizer(text, return_tensors="pt")
        return self.bert(**inputs).last_hidden_state
    
    def _process_image(self, image):
        """Базовая обработка изображений"""
        return image.mean(dim=[2, 3])  # Глобальный пулинг