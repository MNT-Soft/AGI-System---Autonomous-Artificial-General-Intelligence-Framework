import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from config.settings import settings

class NeuralCore(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Инициализация языковой модели
        self.lm = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Адаптивные слои
        self.attention = nn.MultiheadAttention(
            embed_dim=settings.EMBEDDING_SIZE,
            num_heads=4
        )
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(settings.EMBEDDING_SIZE, settings.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(settings.HIDDEN_SIZE, settings.HIDDEN_SIZE//2)
        ])
        
    def forward(self, inputs):
        if isinstance(inputs, str):
            inputs = self.tokenizer(inputs, return_tensors="pt")
            
        # Получение эмбеддингов
        embeddings = self.lm(**inputs).last_hidden_state
        
        # Механизм внимания
        attn_output, _ = self.attention(embeddings, embeddings, embeddings)
        
        # Полносвязные слои
        for layer in self.fc_layers:
            attn_output = layer(attn_output)
            
        return attn_output
    
    def evolve(self, complexity):
        """Адаптация архитектуры под сложность задач"""
        if complexity > 0.7 and len(self.fc_layers) < 5:
            new_layer = nn.Linear(settings.HIDDEN_SIZE//2, settings.HIDDEN_SIZE//4)
            self.fc_layers.append(new_layer)
            self.fc_layers.append(nn.ReLU())
            return True
        return False