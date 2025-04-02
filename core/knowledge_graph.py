import networkx as nx
from typing import Dict, List, Optional
from config import config
import json

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.load()
    
    def add_concept(self, concept: str, metadata: Optional[Dict] = None):
        """Добавление новой концепции"""
        if not metadata:
            metadata = {}
            
        if concept not in self.graph:
            self.graph.add_node(concept, **metadata)
    
    def add_relation(self, source: str, target: str, rel_type: str, weight: float = 1.0):
        """Установление связи между концепциями"""
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, relation=rel_type, weight=weight)
    
    def query(self, concept: str, depth: int = 2) -> List[Dict]:
        """Поиск связанных концепций"""
        if concept not in self.graph:
            return []
            
        results = []
        for node in nx.dfs_preorder_nodes(self.graph, concept, depth_limit=depth):
            edges = self.graph.edges(node, data=True)
            results.append({
                "node": node,
                "edges": [(target, data) for _, target, data in edges]
            })
        return results
    
    def infer(self, concept1: str, concept2: str) -> Optional[str]:
        """Логический вывод между концепциями"""
        try:
            path = nx.shortest_path(self.graph, concept1, concept2)
            relations = []
            for i in range(len(path)-1):
                rel = self.graph.edges[path[i], path[i+1]]["relation"]
                relations.append(f"{path[i]} -> {rel} -> {path[i+1]}")
            return " | ".join(relations)
        except nx.NetworkXNoPath:
            return None
    
    def save(self):
        """Сохранение графа в формате JSON"""
        data = {
            "nodes": list(self.graph.nodes(data=True)),
            "edges": list(self.graph.edges(data=True))
        }
        with open(config.KNOWLEDGE_PATH, 'w') as f:
            json.dump(data, f)
    
    def load(self):
        """Загрузка графа из файла"""
        try:
            with open(config.KNOWLEDGE_PATH, 'r') as f:
                data = json.load(f)
                self.graph.add_nodes_from(data["nodes"])
                self.graph.add_edges_from(data["edges"])
        except (FileNotFoundError, json.JSONDecodeError):
            self.graph = nx.DiGraph()