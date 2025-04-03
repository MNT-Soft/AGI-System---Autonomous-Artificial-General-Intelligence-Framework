import sqlite3
import torch
from typing import List, Dict
from config.settings import settings
from utils.logger import setup_logger
import time

logger = setup_logger(__name__)

class DataManager:
    def __init__(self):
        self.conn = sqlite3.connect(settings.DATA_DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT,
                timestamp REAL
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_id INTEGER,
                embedding BLOB,
                FOREIGN KEY (data_id) REFERENCES raw_data(id)
            )
        """)
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON raw_data(timestamp)")
        self.conn.commit()
    
    def store_data(self, content: str, source: str, timestamp: float):
        self.cursor.execute(
            "INSERT INTO raw_data (content, source, timestamp) VALUES (?, ?, ?)",
            (content, source, timestamp)
        )
        data_id = self.cursor.lastrowid
        self.conn.commit()
        return data_id
    
    def store_embedding(self, data_id: int, embedding: torch.Tensor):
        embedding_bytes = embedding.cpu().numpy().tobytes()
        self.cursor.execute(
            "INSERT INTO embeddings (data_id, embedding) VALUES (?, ?)",
            (data_id, embedding_bytes)
        )
        self.conn.commit()
    
    def store_long_term_memory(self, embedding: torch.Tensor, timestamp: float):
        embedding_bytes = embedding.cpu().numpy().tobytes()
        self.cursor.execute(
            "INSERT INTO embeddings (data_id, embedding) VALUES (NULL, ?)",
            (embedding_bytes,)
        )
        self.conn.commit()
    
    def fetch_batch(self, batch_size: int = settings.BATCH_SIZE) -> List[Dict]:
        self.cursor.execute("SELECT id, content FROM raw_data ORDER BY timestamp DESC LIMIT ?", (batch_size,))
        rows = self.cursor.fetchall()
        return [{"id": row[0], "content": row[1]} for row in rows]
    
    def clean_old_data(self, max_age: float = 30 * 24 * 3600):
        threshold = time.time() - max_age
        self.cursor.execute("DELETE FROM raw_data WHERE timestamp < ?", (threshold,))
        self.cursor.execute("DELETE FROM embeddings WHERE data_id NOT IN (SELECT id FROM raw_data)")
        self.conn.commit()
        logger.info(f"Cleaned old data, removed {self.cursor.rowcount} entries")
    
    def close(self):
        self.conn.close()