import time
import logging
import psutil
from typing import Dict, Any
from core.neural_processor import NeuralProcessor
from core.knowledge_graph import KnowledgeGraph
from modules.code_executor import CodeExecutor
from config import config

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
        """Инициализация системы с проверкой ресурсов"""
        self._check_system_resources()
        
        self.brain = NeuralProcessor()
        self.knowledge = KnowledgeGraph()
        self.executor = CodeExecutor()
        
        self.complexity = 0.5
        self.performance = 0.0
        self.safe_mode = False
        
        self._initialize_base_knowledge()
        logger.info("AGI система инициализирована")

    def _check_system_resources(self):
        """Проверка доступных ресурсов перед запуском"""
        mem = psutil.virtual_memory()
        if mem.percent > config.MAX_MEMORY * 100:
            raise RuntimeError(
                f"Недостаточно памяти. Используется {mem.percent}%, "
                f"лимит {config.MAX_MEMORY*100}%"
            )
        
        if not torch.cuda.is_available():
            logger.warning("CUDA не доступна, будет использоваться CPU")

    def _initialize_base_knowledge(self):
        """Инициализация базовых знаний"""
        try:
            base_concepts = {
                "обучение": {"type": "процесс", "timestamp": time.time()},
                "оптимизация": {"type": "процесс", "timestamp": time.time()},
                "данные": {"type": "сущность", "timestamp": time.time()},
                "код": {"type": "сущность", "timestamp": time.time()}
            }
            
            for concept, meta in base_concepts.items():
                self.knowledge.add_concept(concept, meta)
            
            relations = [
                ("обучение", "данные", "использует"),
                ("оптимизация", "код", "генерирует"),
                ("данные", "код", "преобразуется_в")
            ]
            
            for src, tgt, rel in relations:
                self.knowledge.add_relation(src, tgt, rel)
                
        except Exception as e:
            logger.error("Ошибка инициализации знаний: %s", str(e))
            raise RuntimeError("Не удалось инициализировать базовые знания") from e

    def process(self, input_data: str) -> Dict[str, Any]:
        """Основной цикл обработки входных данных"""
        if self.safe_mode:
            return {"status": "error", "message": "Система в безопасном режиме"}
            
        if not isinstance(input_data, str) or len(input_data) > 10000:
            return {"status": "error", "message": "Некорректные входные данные"}
        
        try:
            # 1. Мониторинг ресурсов
            self._monitor_resources()
            
            # 2. Обработка данных
            start_time = time.time()
            processed = self.brain(input_data)
            analysis_time = time.time() - start_time
            
            # 3. Обновление знаний
            self.knowledge.add_concept(input_data[:50], 
                                     {"timestamp": time.time()})
            
            # 4. Планирование действия
            action = self._plan_action(processed)
            
            # 5. Выполнение действия
            result = self._execute_action(action)
            
            # 6. Адаптация
            self._adapt(analysis_time)
            
            return {
                "status": "success",
                "action": action["type"],
                "result": result,
                "complexity": self.complexity,
                "performance": self.performance,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error("Ошибка обработки: %s", str(e), exc_info=True)
            self._enter_safe_mode()
            return {
                "status": "error",
                "message": str(e),
                "timestamp": time.time()
            }

    def _monitor_resources(self):
        """Мониторинг использования ресурсов"""
        mem = psutil.virtual_memory()
        if mem.percent > config.MAX_MEMORY * 100:
            logger.warning(
                "Превышение использования памяти: %.1f%% > %.1f%%", 
                mem.percent, config.MAX_MEMORY*100
            )
            self._free_up_memory()
            
        if psutil.cpu_percent() > 90:
            logger.warning("Высокая загрузка CPU: %.1f%%", psutil.cpu_percent())

    def _free_up_memory(self):
        """Освобождение памяти"""
        logger.info("Освобождение памяти...")
        torch.cuda.empty_cache()
        self.complexity = max(0.1, self.complexity - 0.1)
        
        if psutil.virtual_memory().percent > config.MAX_MEMORY * 100:
            self._enter_safe_mode()

    def _enter_safe_mode(self):
        """Переход в безопасный режим"""
        self.safe_mode = True
        logger.critical("АКТИВИРОВАН БЕЗОПАСНЫЙ РЕЖИМ")
        
        # Сохранение состояния
        try:
            self.brain.save(config.MODEL_PATH)
            self.knowledge.save()
        except Exception as e:
            logger.error("Ошибка сохранения состояния: %s", str(e))

    def _plan_action(self, processed_data) -> Dict[str, Any]:
        """Планирование действия на основе текущего состояния"""
        try:
            if self.complexity > 0.7 and self.performance > 0.6:
                return {
                    "type": "code",
                    "task": "optimization",
                    "params": {"size": min(1000000, int(1000 * self.complexity))}
                }
                
            return {
                "type": "learn",
                "task": "process_data",
                "params": {"data": processed_data}
            }
        except Exception as e:
            logger.error("Ошибка планирования: %s", str(e))
            return {
                "type": "wait",
                "task": "recover",
                "params": {}
            }

    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение запланированного действия"""
        try:
            if action["type"] == "code":
                success, output = self.executor.execute(
                    action["task"],
                    "python",
                    action["params"]
                )
                return {
                    "action": "code_execution",
                    "success": success,
                    "output": output
                }
            elif action["type"] == "learn":
                return {
                    "action": "learning",
                    "status": "completed"
                }
            else:
                return {
                    "action": "no_op",
                    "status": "completed"
                }
        except Exception as e:
            logger.error("Ошибка выполнения: %s", str(e))
            return {
                "action": action["type"],
                "status": "failed",
                "error": str(e)
            }

    def _adapt(self, last_execution_time: float):
        """Адаптация системы на основе производительности"""
        try:
            # Обновление метрик
            self.performance = 1.0 / (last_execution_time + 0.1)
            self.complexity = min(1.0, self.complexity + 0.01)
            
            # Эволюция архитектуры
            if self.brain.evolve(self.complexity):
                logger.info("Архитектура нейросети адаптирована")
                
            # Сохранение состояния
            if time.time() % 300 < 0.1:  # Каждые ~5 минут
                self.brain.save(config.MODEL_PATH)
                self.knowledge.save()
                
        except Exception as e:
            logger.error("Ошибка адаптации: %s", str(e))
            self.complexity = max(0.1, self.complexity - 0.2)

def main():
    """Точка входа в систему"""
    try:
        system = AGISystem()
        
        while True:
            try:
                user_input = input("\nВвод (или 'exit' для выхода): ").strip()
                if user_input.lower() == 'exit':
                    break
                    
                if not user_input:
                    continue
                    
                result = system.process(user_input)
                print(f"\nРезультат: {result}")
                
            except KeyboardInterrupt:
                print("\nЗавершение работы...")
                break
            except Exception as e:
                print(f"\nОшибка: {str(e)}")
                continue
                
    except Exception as e:
        logger.critical("Критическая ошибка инициализации: %s", str(e), exc_info=True)
        print(f"Системная ошибка: {str(e)}")
    finally:
        logger.info("Система завершила работу")
        print("Работа системы завершена")

if __name__ == "__main__":
    main()
