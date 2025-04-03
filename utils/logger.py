import logging
import sys
from logging.handlers import RotatingFileHandler
from config.settings import settings

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if settings.LOG_FILE:
        try:
            fh = RotatingFileHandler(settings.LOG_FILE, maxBytes=10_000_000, backupCount=5)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            logger.error(f"Failed to setup file logging: {e}")
    
    return logger