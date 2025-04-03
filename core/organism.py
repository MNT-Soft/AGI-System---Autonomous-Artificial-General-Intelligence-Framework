import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel
from config.settings import settings
from typing import Dict, List, Optional
from torch.optim import Adam
from torch.nn.parallel import DataParallel
from collections import deque
from utils.logger import setup_logger
import random
import numpy as np
from torchvision import models

logger = setup_logger(__name__)

class Organism(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_parallel = torch.cuda.device_count() > 1
        
        self.sensory = nn.ModuleDict({
            "text": AutoModel.from_pretrained("bert-base-uncased"),
            "image": CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
            "vision": models.resnet50(pretrained=True)
        })
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.brain = nn.ModuleDict({
            "attention": nn.ModuleList([nn.MultiheadAttention(embed_dim=settings.EMBEDDING_SIZE, num_heads=8)]),
            "processing": nn.ModuleList([nn.Linear(settings.EMBEDDING_SIZE, settings.HIDDEN_SIZE), nn.ReLU(), nn.Dropout(0.1)]),
            "consciousness": nn.Sequential(nn.Linear(settings.HIDDEN_SIZE, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()),
            "generator": nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=settings.HIDDEN_SIZE, nhead=8), num_layers=3
            )
        })
        
        self.short_term_memory = deque(maxlen=100)
        self.long_term_memory = []
        self.emotions = {"curiosity": 0.5, "satisfaction": 0.5, "fear": 0.2, "joy": 0.5, "anger": 0.2}
        
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.meta_optimizer = Adam(self.parameters(), lr=settings.META_LR)
        self.criterion = nn.MSELoss()
        
        if self.is_parallel:
            for key in self.sensory:
                self.sensory[key] = DataParallel(self.sensory[key])
            self.brain["attention"] = nn.ModuleList([DataParallel(layer) for layer in self.brain["attention"]])
            self.brain["processing"] = nn.ModuleList([DataParallel(layer) for layer in self.brain["processing"]])
            self.brain["consciousness"] = DataParallel(self.brain["consciousness"])
            self.brain["generator"] = DataParallel(self.brain["generator"])
        self.to(self.device)
    
    def perceive(self, inputs, modality: str = "text") -> Dict:
        try:
            if modality == "text":
                if isinstance(inputs, str):
                    inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                elif isinstance(inputs, list):
                    inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                sensory_output = self.sensory["text"](**inputs).last_hidden_state
            elif modality == "image":
                inputs = self.clip_processor(images=inputs, return_tensors="pt").to(self.device)
                sensory_output = self.sensory["image"].get_image_features(**inputs)
                sensory_output = sensory_output.unsqueeze(1)
            
            attn_output = sensory_output
            for attn_layer in self.brain["attention"]:
                attn_output, attn_weights = attn_layer(attn_output, attn_output, attn_output)
            
            output = attn_output
            for layer in self.brain["processing"]:
                output = layer(output)
            
            awareness = self.brain["consciousness"](output.mean(dim=1))
            self.short_term_memory.append(output.mean(dim=1).detach())
            if self.emotions["satisfaction"] > 0.7:
                self._consolidate_memory(output.mean(dim=1))
            self._update_emotions(awareness)
            return {"output": output, "awareness": awareness, "attention_weights": attn_weights}
        except Exception as e:
            logger.error(f"Perception error: {e}")
            return {}
    
    def think(self, perception: Dict) -> Dict:
        if not perception:
            return {}
        output = perception["output"]
        awareness = perception["awareness"].mean().item()
        return {
            "decision": self._decide_action(awareness),
            "complexity": self._assess_complexity(),
            "query": self._generate_query() if self.emotions["curiosity"] > 0.7 else None
        }
    
    def learn(self, inputs, reward: float, modality: str = "text"):
        try:
            result = self.perceive(inputs, modality)
            if not result:
                return float("inf")
            output = result["output"].mean(dim=1)
            target = torch.full_like(output, reward, device=self.device)
            
            self.optimizer.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            if self.emotions["curiosity"] > 0.8:
                self._meta_learn(inputs, modality)
            
            return loss.item()
        except Exception as e:
            logger.error(f"Learning error: {e}")
            return float("inf")
    
    def _meta_learn(self, inputs, modality):
        self.meta_optimizer.zero_grad()
        meta_loss = 0
        for _ in range(2):
            result = self.perceive(inputs, modality)
            output = result["output"].mean(dim=1)
            meta_loss += self.criterion(output, torch.ones_like(output))
        meta_loss.backward()
        self.meta_optimizer.step()
    
    def generate(self, context: torch.Tensor) -> str:
        try:
            if context.dim() < 2:
                context = context.unsqueeze(0)
            output = self.brain["generator"](context.to(self.device), torch.zeros_like(context))
            tokens = torch.argmax(output, dim=-1)
            return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""
    
    def compress(self):
        for layer in self.brain["processing"]:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data
                mask = (weight.abs() > 0.01).float()
                layer.weight.data *= mask
        logger.info("Model compressed")
    
    def repair(self):
        if self.short_term_memory:
            variance = torch.var(torch.stack(list(self.short_term_memory)), dim=0).mean().item()
            if variance > 10:
                self.brain["processing"] = self.brain["processing"][:-2]
                logger.info("Repaired: Removed unstable layers")
    
    def check_resources(self):
        if torch.cuda.is_available() and torch.cuda.memory_allocated() / torch.cuda.max_memory() > settings.MAX_MEMORY_USAGE:
            self.compress()
            logger.info("Compressed model due to high memory usage")
    
    def introspect(self) -> Dict:
        if not self.short_term_memory:
            return {"status": "No memory available"}
        
        short_term = torch.stack(list(self.short_term_memory))
        long_term = torch.stack(self.long_term_memory) if self.long_term_memory else torch.tensor([]).to(self.device)
        short_term_variance = torch.var(short_term, dim=0).mean().item()
        long_term_variance = torch.var(long_term, dim=0).mean().item() if long_term.numel() > 0 else 0
        
        return {
            "short_term_variance": short_term_variance,
            "long_term_variance": long_term_variance,
            "complexity": self._assess_complexity(),
            "emotions": self.emotions.copy(),
            "recommendation": self._introspection_recommendation(short_term_variance, self._assess_complexity())
        }
    
    def prune_memory(self, threshold: float = 0.1):
        self.short_term_memory = deque([m for m in self.short_term_memory if torch.norm(m) > threshold], maxlen=100)
        logger.info(f"Short-term memory pruned, size={len(self.short_term_memory)}")
    
    def _consolidate_memory(self, memory_vector):
        compressed = torch.mean(torch.stack(list(self.short_term_memory)), dim=0)
        self.long_term_memory.append(compressed.detach())
        logger.info(f"Consolidated memory, long-term size={len(self.long_term_memory)}")
    
    def _assess_complexity(self) -> float:
        return sum(len(self.brain[k]) if isinstance(self.brain[k], nn.ModuleList) else 1 for k in self.brain)
    
    def _decide_action(self, awareness: float) -> str:
        if self.emotions["curiosity"] > 0.7 or awareness < 0.3:
            return "explore"
        elif self.emotions["joy"] > 0.6:
            return "create"
        elif self.emotions["anger"] > 0.5:
            return "repair"
        return "learn"
    
    def _update_emotions(self, awareness: torch.Tensor):
        avg_awareness = awareness.mean().item()
        self.emotions["curiosity"] = min(1.0, self.emotions["curiosity"] + (1 - avg_awareness) * 0.1)
        self.emotions["satisfaction"] = min(1.0, self.emotions["satisfaction"] + avg_awareness * 0.1)
        self.emotions["fear"] = max(0.0, self.emotions["fear"] - avg_awareness * 0.05)
        self.emotions["joy"] = min(1.0, self.emotions["joy"] + (avg_awareness - self.emotions["fear"]) * 0.1)
        self.emotions["anger"] = max(0.0, self.emotions["anger"] + (self.emotions["fear"] - avg_awareness) * 0.05)
    
    def _generate_query(self) -> str:
        concepts = list(self.long_term_memory[-10:]) if self.long_term_memory else [torch.zeros(settings.HIDDEN_SIZE).to(self.device)]
        return self.generate(torch.stack(concepts)) or "What is the latest in AI research?"
    
    def _introspection_recommendation(self, variance: float, complexity: float) -> str:
        if variance < 0.01:
            return "Seek new experiences."
        elif complexity > 50:
            return "Optimize resource use."
        return "Balanced state."
    
    def decide_self_modification(self, desires: Dict) -> Optional[Dict]:
        if desires["complexity"] > 0.9:
            return self.evolve(trigger_code_evolution=True)
        elif desires["efficiency"] > 0.9:
            return {"method": "compress", "params": {}}
        return None
    
    def save_state(self, path: str = settings.MODEL_PATH):
        torch.save(self.state_dict(), path)
    
    def load_state(self, path: str = settings.MODEL_PATH):
        try:
            self.load_state_dict(torch.load(path))
            self.to(self.device)
        except FileNotFoundError:
            logger.info(f"No state found at {path}, starting fresh.")
    
    def evolve(self, trigger_code_evolution: bool = False):
        current_dim = self.brain["processing"][-1].out_features
        new_dim = current_dim * 2 if self.emotions["curiosity"] > 0.7 else current_dim // 2
        new_layer = nn.Linear(current_dim, max(new_dim, 128)).to(self.device)
        self.brain["processing"].append(new_layer)
        self.brain["processing"].append(nn.ReLU())
        
        if self.emotions["joy"] > 0.8:
            self.brain["new_organ"] = nn.Linear(current_dim, 256).to(self.device)
        
        if self.is_parallel:
            self.brain["processing"] = nn.ModuleList([DataParallel(layer) for layer in self.brain["processing"]])
            if "new_organ" in self.brain:
                self.brain["new_organ"] = DataParallel(self.brain["new_organ"])
        self.optimizer = Adam(self.parameters(), lr=0.001)
        logger.info(f"Evolved: Added new layer/organ, complexity={self._assess_complexity()}")
        
        if trigger_code_evolution:
            return {"method": "add_layer", "params": {"in_features": current_dim, "out_features": new_dim}}
        return None