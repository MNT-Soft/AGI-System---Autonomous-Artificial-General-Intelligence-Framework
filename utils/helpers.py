from utils.logger import setup_logger

logger = setup_logger(__name__)

def format_time(seconds: float) -> str:
    """Format time in seconds to a readable string."""
    return f"{seconds:.2f}s"