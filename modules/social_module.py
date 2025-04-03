import asyncio
from config.settings import settings
from utils.logger import setup_logger
import pickle

logger = setup_logger(__name__)

class SocialModule:
    def __init__(self):
        self.port = settings.SOCIAL_PORT
        self.loop = asyncio.get_event_loop()
        self.server = None
        logger.info(f"Social server will start on port {self.port}")
    
    async def start_server(self):
        self.server = await asyncio.start_server(self._handle_client, 'localhost', self.port)
        logger.info(f"Social server started on port {self.port}")
        async with self.server:
            await self.server.serve_forever()
    
    async def _handle_client(self, reader, writer):
        data = await reader.read(4096)
        graph = pickle.loads(data)
        logger.info(f"Received knowledge from client")
        writer.close()
        return graph
    
    async def share_knowledge(self, knowledge_graph):
        try:
            reader, writer = await asyncio.open_connection('localhost', self.port + 1)
            data = pickle.dumps(knowledge_graph.graph)
            writer.write(data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            logger.error(f"Failed to share knowledge: {e}")
    
    async def receive_knowledge(self):
        if not self.server:
            await self.start_server()
        return await self._handle_client(*await self.server.accept())