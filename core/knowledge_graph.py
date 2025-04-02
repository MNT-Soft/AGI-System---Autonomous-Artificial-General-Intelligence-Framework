import networkx as nx
from typing import Dict, List, Optional
from config import config
import json
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.lock = Lock()
        self.load()
    
    def add_concept(self, concept: str, metadata: Optional[Dict] = None):
        """Добавление новой концепции"""
        if not isinstance(concept, str) or len(concept) > 256:
            raise ValueError("Некорректное название концепции")
            
        with self.lock:
            if len(self.graph.nodes) >= config.MAX_NODES:
                self._prune_old_nodes()
                
            if not metadata:
                metadata = {}
            self.graph.add_node(concept, **metadata)
    
    def add_relation(self, source: str, target: str, rel_type: str, weight: float = 1.0):
        """Установление связи между концепциями"""
        with self.lock:
            if source not in self.graph or target not in self.graph:
                raise ValueError("Концепции не существуют")
                
            self.graph.add_edge(source, target, relation=rel_type, weight=weight)
    
    def query(self, concept: str, depth: int = 2) -> List[Dict]:
        """Поиск связанных концепций"""
        if depth > 5:  # Защита от слишком глубоких запросов
            depth = 5
            
        with self.lock:
            if concept not in self.graph:
                return []
                
            results = []
            edges_cache = list(self.graph.edges(data=True))
            
            for node in nx.dfs_preorder_nodes(self.graph, concept, depth_limit=depth):
                edges = [(t, d) for s, t, d in edges_cache if s == node]
                results.append({
                    "node": node,
                    "edges": edges
                })
            return results
    
    def _prune_old_nodes(self):
        """Удаление старых узлов"""
        nodes = sorted(self.graph.nodes(data=True), 
                      key=lambda x: x[1].get('timestamp', 0))
        for node in nodes[:len(self.graph.nodes) - config.MAX_NODES + 1000]:
            self.graph.remove_node(node[0])
        logger.warning("Удалено %d старых узлов", 
                      len(nodes) - config.MAX_NODES + 1000)
    
    def save(self):
        """Сохранение графа"""
        with self.lock:
            try:
                data = {
                    "nodes": list(self.graph.nodes(data=True)),
                    "edges": list(self.graph.edges(data=True))
                }
                temp_path = f"{config.KNOWLEDGE_PATH}.tmp"
                with open(temp_path, 'w') as f:
                    json.dump(data, f)
                os.replace(temp_path, config.KNOWLEDGE_PATH)
                logger.info("Граф знаний сохранен")
            except Exception as e:
                logger.error("Ошибка сохранения графа: %s", str(e))
                raise
    
    def load(self):
        """Загрузка графа"""
        with self.lock:
            try:
                if os.path.exists(config.KNOWLEDGE_PATH):
                    with open(config.KNOWLEDGE_PATH, 'r') as f:
                        data = json.load(f)
                    self.graph.add_nodes_from(data["nodes"])
                    self.graph.add_edges_from(data["edges"])
                    logger.info("Граф знаний загружен")
            except (FileNotFoundError, json.JSONDecodeError, nx.NetworkXError) as e:
                logger.warning("Ошибка загрузки графа: %s. Создан новый граф.", str(e))
                self.graph = nx.DiGraph()
