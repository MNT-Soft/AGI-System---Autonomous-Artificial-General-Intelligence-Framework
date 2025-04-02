import networkx as nx
from typing import Optional, Dict, List

class KnowledgeGraph:
    def __init__(self, config):
        self.config = config
        self.graph = nx.DiGraph()
        self.load()
        
    def add_concept(self, concept: str, relations: Optional[Dict] = None):
        """Добавление новой концепции"""
        if concept not in self.graph:
            self.graph.add_node(concept)
            
        if relations:
            for target, rel_type in relations.items():
                self.graph.add_edge(concept, target, relation=rel_type)
                
    def query(self, concept: str, depth: int = 1) -> List[str]:
        """Поиск связанных концепций"""
        if concept not in self.graph:
            return []
            
        return list(nx.dfs_preorder_nodes(self.graph, concept, depth_limit=depth))
    
    def save(self):
        """Сохранение графа"""
        nx.write_gml(self.graph, self.config.KNOWLEDGE_GRAPH)
        
    def load(self):
        """Загрузка графа"""
        if os.path.exists(self.config.KNOWLEDGE_GRAPH):
            self.graph = nx.read_gml(self.config.KNOWLEDGE_GRAPH)