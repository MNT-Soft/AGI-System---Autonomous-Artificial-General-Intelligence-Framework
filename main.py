import time
from typing import Dict, Any
from core.neural_processor import NeuralProcessor
from core.knowledge_graph import KnowledgeGraph
from modules.code_executor import CodeExecutor
from config import config
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AGI-System")

class AGISystem:
    def __init__(self):
        self.brain = NeuralProcessor()
        self.knowledge = KnowledgeGraph()
        self.executor = CodeExecutor()
        self.complexity = 0.5
        self.performance = 0.0
        
        # Инициализация базовых знаний
        self._initialize_base_knowledge()
    
    def _initialize_base_knowledge(self):
        """Инициализация базовых концепций"""
        base_concepts = {
            "learning": {"type": "process"},
            "optimization": {"type": "process"},
            "data": {"type": "entity"},
            "code": {"type": "entity"}
        }
        
        for concept, meta in base_concepts.items():
            self.knowledge.add_concept(concept, meta)
        
        relations = [
            ("learning", "data", "requires"),
            ("optimization", "code", "generates"),
            ("data", "code", "can_transform")
        ]
        
        for src, tgt, rel in relations:
            self.knowledge.add_relation(src, tgt, rel)
    
    def process(self, input_data: str) -> Dict[str, Any]:
        """Основной цикл обработки входных данных"""
        try:
            # 1. Анализ входных данных
            analysis_start = time.time()
            processed = self.brain(input_data)
            analysis_time = time.time() - analysis_start
            
            # 2. Обновление знаний
            self.knowledge.add_concept(input_data[:50])
            
            # 3. Планирование действия
            action = self._plan_action(processed)
            
            # 4. Выполнение действия
            result = self._execute_action(action)
            
            # 5. Адаптация
            self._adapt(analysis_time)
            
            return {
                "status": "success",
                "action": action,
                "result": result,
                "complexity": self.complexity,
                "performance": self.performance
            }
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _plan_action(self, processed_data) -> Dict[str, Any]:
        """Планирование действия на основе текущего состояния"""
        if self.complexity > 0.7 and self.performance > 0.6:
            return {
                "type": "code",
                "task": "optimization",
                "params": {"size": int(1000 * self.complexity)}
            }
        return {
            "type": "learn",
            "task": "process_data",
            "params": {"data": processed_data}
        }
    
    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение запланированного действия"""
        if action["type"] == "code":
            success, output = self.executor.execute(
                action["task"],
                "python",
                action["params"]
            )
            return {"code_execution": {"success": success, "output": output}}
        
        return {"learning": {"status": "completed"}}
    
    def _adapt(self, last_execution_time: float):
        """Адаптация системы на основе производительности"""
        # Обновление сложности
        self.performance = 1.0 / (last_execution_time + 0.1)
        self.complexity = min(1.0, self.complexity + 0.01)
        
        # Эволюция архитектуры
        if self.brain.evolve(self.complexity):
            logger.info("Neural architecture evolved")
        
        # Сохранение состояния
        self.brain.save(config.MODEL_PATH)
        self.knowledge.save()

def main():
    system = AGISystem()
    logger.info("AGI System initialized")
    
    try:
        while True:
            user_input = input("Enter input (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
                
            result = system.process(user_input)
            print(f"Result: {result}")
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        logger.info("System shutdown completed")

if __name__ == "__main__":
    main()