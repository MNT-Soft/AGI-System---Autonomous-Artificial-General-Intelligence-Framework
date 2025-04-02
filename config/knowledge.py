import networkx as nx
from typing import Dict, List
from config.settings import settings

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.filepath = settings.KNOWLEDGE_GRAPH_PATH
        
    def add_node(self, concept: str, metadata: Dict = None):
        """Добавление концепции в граф"""
        self.graph.add_node(concept, **metadata if metadata else {})
        
    def add_relation(self, source: str, target: str, relation: str):
        """Установление связи между концепциями"""
        self.graph.add_edge(source, target, relation=relation)
        
    def query(self, concept: str, depth: int = 2) -> List[Dict]:
        """Запрос связанных концепций"""
        if concept not in self.graph:
            return []
            
        results = []
        for node in nx.dfs_preorder_nodes(self.graph, concept, depth_limit=depth):
            neighbors = list(self.graph.neighbors(node))
            results.append({
                "concept": node,
                "relations": [self.graph.edges[node, n]["relation"] for n in neighbors]
            })
        return results
    
    def save(self):
        """Сохранение графа на диск"""
        nx.write_gml(self.graph, self.filepath)
        
    def load(self):
        """Загрузка графа с диска"""
        try:
            self.graph = nx.read_gml(self.filepath)
        except FileNotFoundError:
            self.graph = nx.DiGraph()