from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    def load(self, source: str) -> list:
        try:
            with open(source, 'r') as f:
                return f.readlines()
        except Exception as e:
            logger.error(f"Data load error: {e}")
            return []