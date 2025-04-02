import psutil
import torch

def check_resources(config):
    """Проверка использования ресурсов"""
    mem = psutil.virtual_memory()
    if mem.percent > config.MAX_MEMORY_USAGE * 100:
        raise RuntimeError(f"Memory usage exceeded {config.MAX_MEMORY_USAGE*100}%")
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def safe_execute(func, *args, **kwargs):
    """Безопасное выполнение с обработкой ошибок"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return f"Error: {str(e)}"