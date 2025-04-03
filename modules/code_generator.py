from core.executor import CodeExecutor
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CodeExecutor:
    def execute(self, code_type: str, params: dict):
        try:
            if code_type == "optimization":
                size = params.get("size", 1000)
                result = [i * i for i in range(size)]
                return True, f"Optimized {size} elements"
            return False, "Unknown code type"
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False, str(e)