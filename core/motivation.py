from typing import Dict
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MotivationSystem:
    def __init__(self):
        self.desires = {
            "knowledge": 0.5,
            "complexity": 0.5,
            "efficiency": 0.5,
            "exploration": 0.5
        }
        self.weights = {k: 1.0 for k in self.desires}
    
    def update_desires(self, introspection: Dict, performance: Dict, emotions: Dict = None):
        variance = introspection["memory_variance"]
        complexity = introspection["complexity"]
        loss = performance.get("loss", 0.0)
        
        if variance < 0.01:
            self.desires["exploration"] = min(1.0, self.desires["exploration"] + 0.1)
            self.desires["knowledge"] = min(1.0, self.desires["knowledge"] + 0.05)
        if complexity * 1e6 < introspection["complexity"]:
            self.desires["efficiency"] = min(1.0, self.desires["efficiency"] + 0.1)
        if loss < 0.1 and variance > 0.5:
            self.desires["complexity"] = min(1.0, self.desires["complexity"] + 0.1)
        
        if emotions:
            self.desires["exploration"] += emotions["curiosity"] * 0.1
            self.desires["efficiency"] -= emotions["fear"] * 0.05
        
        total = sum(self.desires.values()) or 1
        for key in self.desires:
            self.desires[key] = min(1.0, max(0.0, self.desires[key] / total))
        
        logger.info(f"Updated desires: {self.desires}")
    
    def prioritize_action(self) -> str:
        weighted = {k: v * self.weights[k] for k, v in self.desires.items()}
        return max(weighted, key=weighted.get)