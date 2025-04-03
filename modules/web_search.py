import aiohttp
import time
from config.settings import settings
from utils.logger import setup_logger
from hashlib import md5

logger = setup_logger(__name__)

class WebSearch:
    def __init__(self):
        self.api_key = settings.GOOGLE_API_KEY
        self.cse_id = settings.GOOGLE_CSE_ID
        self.search_url = "https://www.googleapis.com/customsearch/v1"
        self.seen_hashes = set()
    
    async def search(self, query: str, num_results: int = 10, start: int = 1, session=None):
        if not self.api_key or not self.cse_id:
            logger.error("Google API key or CSE ID missing")
            return []
        try:
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": min(num_results, 10),
                "start": start
            }
            async with session.get(self.search_url, params=params) as response:
                response.raise_for_status()
                results = (await response.json()).get("items", [])
                filtered = []
                for item in results:
                    content_hash = md5(item["snippet"].encode()).hexdigest()
                    if content_hash not in self.seen_hashes:
                        self.seen_hashes.add(content_hash)
                        filtered.append({"snippet": item["snippet"], "link": item["link"]})
                return filtered
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def async_bulk_search(self, query: str, total_results: int = 100, session=None):
        if not session:
            async with aiohttp.ClientSession() as session:
                return await self.async_bulk_search(query, total_results, session)
        results = []
        for start in range(1, total_results + 1, 10):
            batch = await self.search(query, num_results=10, start=start, session=session)
            results.extend(batch)
            await asyncio.sleep(1)
            if len(batch) < 10:
                break
        return results[:total_results]