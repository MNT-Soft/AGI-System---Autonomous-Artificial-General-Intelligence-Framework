from utils.logger import setup_logger

logger = setup_logger(__name__)

class Monitor:
    def __init__(self):
        self.metrics = {}
    
    def log_metric(self, name: str, value: float):
        self.metrics[name] = value
        logger.info(f"Metric {name}: {value}")
    
    def get_metrics(self) -> dict:
        return self.metrics.copy()