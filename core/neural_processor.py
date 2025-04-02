import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from config import config

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
        if isinstance(inputs, str):
            inputs = self.tokenizer(inputs, return_tensors="pt")
        
        embeddings = self.encoder(**inputs).last_hidden_state
        attn_out, _ = self.attention(embeddings, embeddings, embeddings)
        return self.projection(attn_out.mean(dim=1))
    
    def evolve(self, current_complexity):
        """Адаптация архитектуры"""
        if (current_complexity > self.complexity_threshold and 
            self.current_layers < self.max_layers):
            
            new_layer = nn.Sequential(
                nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE),
                nn.GELU(),
                nn.LayerNorm(config.HIDDEN_SIZE)
            )
            
            # Динамическое добавление слоя
            self.projection.add_module(f"layer_{self.current_layers}", new_layer)
            self.current_layers += 1
            return True
        return False
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))