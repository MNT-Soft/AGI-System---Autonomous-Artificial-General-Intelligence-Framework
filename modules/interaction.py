import requests
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

class WorldInteraction:
    def __init__(self):
        self.openai_api_key = settings.OPENAI_API_KEY
        self.news_api_key = settings.NEWS_API_KEY
    
    def communicate(self, message: str) -> str:
        if not self.openai_api_key:
            logger.error("OpenAI API key missing")
            return ""
        try:
            headers = {"Authorization": f"Bearer {self.openai_api_key}"}
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": message}]
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Communication failed: {e}")
            return ""
    
    def sense_world(self) -> List[str]:
        if not self.news_api_key:
            logger.error("News API key missing")
            return []
        try:
            response = requests.get(f"https://newsapi.org/v2/top-headlines?country=us&apiKey={self.news_api_key}")
            response.raise_for_status()
            return [article["title"] for article in response.json()["articles"]]
        except Exception as e:
            logger.error(f"World sensing failed: {e}")
            return []