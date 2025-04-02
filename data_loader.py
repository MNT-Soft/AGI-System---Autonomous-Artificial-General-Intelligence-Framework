import requests
import cv2
import numpy as np
from typing import Union
from pathlib import Path

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.cache = {}
        
    def load(self, source: str) -> Union[str, np.ndarray]:
        """Универсальный загрузчик данных"""
        if source in self.cache:
            return self.cache[source]
            
        if source.startswith(('http://', 'https://')):
            data = self._fetch_web_data(source)
        else:
            data = self._load_local_file(source)
            
        self.cache[source] = data
        return data
    
    def _load_local_file(self, path: str):
        """Загрузка локальных файлов"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")
            
        if path.suffix in ['.jpg', '.png']:
            return cv2.imread(str(path))
        elif path.suffix in ['.txt', '.csv', '.json']:
            return path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def _fetch_web_data(self, url: str) -> str:
        """Получение данных из интернета"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            raise ConnectionError(f"Failed to fetch {url}: {str(e)}")