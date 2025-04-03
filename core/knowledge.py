import networkx as nx
from typing import Dict, List
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.filepath = settings.KNOWLEDGE_GRAPH_PATH
        self.concept_index = {}
    
    def add_node(self, concept: str, metadata: Dict = None):
        if concept not in self.graph:
            self.graph.add_node(concept, **(metadata if metadata else {}))
            self.concept_index[concept] = metadata or {}
        else:
            if metadata:
                self.graph.nodes[concept].update(metadata)
                self.concept_index[concept].update(metadata)
    
    def add_relation(self, source: str, target: str, relation: str):
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, relation=relation)
    
    def query(self, concept: str, depth: int = 2) -> List[Dict]:
        if concept not in self.concept_index:
            return []
        results = []
        for node in nx.dfs_preorder_nodes(self.graph, concept, depth_limit=depth):
            neighbors = list(self.graph.neighbors(node))
            results.append({
                "concept": node,
                "relations": [self.graph.edges[node, n]["relation"] for n in neighbors],
                "metadata": self.concept_index[node]
            })
        return results
    
    def save(self):
        try:
            nx.write_gml(self.graph, self.filepath)
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
    
    def load(self):
        try:
            self.graph = nx.read_gml(self.filepath)
            self.concept_index = {n: d for n, d in self.graph.nodes(data=True)}
        except FileNotFoundError:
            self.graph = nx.DiGraph()
            self.concept_index = {}