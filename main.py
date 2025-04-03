import asyncio
import time
from core.organism import Organism
from core.knowledge import KnowledgeGraph
from core.motivation import MotivationSystem
from modules.code_generator import CodeExecutor
from modules.code_evolver import CodeEvolver
from modules.web_search import WebSearch
from modules.data_manager import DataManager
from modules.interaction import WorldInteraction
from modules.genetic_evolver import GeneticEvolver
from modules.social_module import SocialModule
from modules.physical_interface import PhysicalInterface
from config.settings import settings
from utils.logger import setup_logger
import torch
import aiohttp

logger = setup_logger(__name__)

class LivingSystem:
    def __init__(self):
        self.organism = Organism()
        self.knowledge = KnowledgeGraph()
        self.motivation = MotivationSystem()
        self.executor = CodeExecutor()
        self.code_evolver = CodeEvolver(target_file="core/organism.py")
        self.web_search = WebSearch()
        self.data_manager = DataManager()
        self.interaction = WorldInteraction()
        self.genetic_evolver = GeneticEvolver(self.organism)
        self.social_module = SocialModule()
        self.physical_interface = PhysicalInterface()
        self.complexity = 0.5
        self.performance_metrics = {"loss": 0.0, "reward": 0.0}
        self.running = True
        self.lock = asyncio.Lock()
        
    async def live(self, stimulus, modality="text", depth=0):
        if depth > 5:
            return False, "Recursion limit reached"
        async with self.lock:
            try:
                perception = self.organism.perceive(stimulus, modality)
                if not perception:
                    return False, "Perception failed"
                
                awareness = perception["awareness"].mean().item()
                logger.info(f"Awareness: {awareness:.3f}, Emotions: {self.organism.emotions}")
                
                self._update_knowledge(stimulus, perception["output"], awareness)
                introspection = self.organism.introspect()
                self.motivation.update_desires(introspection, self.performance_metrics, self.organism.emotions)
                
                thought = self.organism.think(perception)
                action = thought["decision"]
                result = await self._execute_action(action, thought["query"], depth)
                
                self._adapt(introspection, thought["complexity"])
                self._self_evolve()
                
                reward = 1.0 if result[0] else -0.1
                loss = self.organism.learn(stimulus, reward, modality)
                self.performance_metrics["loss"] = loss
                self.performance_metrics["reward"] += reward
                logger.info(f"Loss: {loss:.4f}, Reward: {self.performance_metrics['reward']:.2f}")
                
                return result
                
            except Exception as e:
                logger.error(f"Life cycle error: {e}")
                return False, str(e)
    
    def _update_knowledge(self, stimulus, embeddings, awareness):
        if isinstance(stimulus, str):
            concept = stimulus[:50]
            data_id = self.data_manager.store_data(stimulus, "external", time.time())
            self.data_manager.store_embedding(data_id, embeddings.mean(dim=1))
        else:
            for i, content in enumerate(stimulus):
                concept = content[:50] if isinstance(content, str) else f"image_{i}"
                data_id = self.data_manager.store_data(str(content), "web", time.time())
                self.data_manager.store_embedding(data_id, embeddings[i])
        
        metadata = {"awareness": float(awareness), "embedding": embeddings.mean(dim=0).detach().cpu().numpy()}
        self.knowledge.add_node(concept, metadata)
        if len(self.knowledge.graph.nodes) > 1:
            last_node = list(self.knowledge.graph.nodes)[-2]
            self.knowledge.add_relation(last_node, concept, "related_to")
    
    async def _execute_action(self, action, query, depth):
        if action == "explore":
            async with aiohttp.ClientSession() as session:
                results = await self.web_search.async_bulk_search(query or "latest discoveries", session=session)
            contents = [r["snippet"] for r in results]
            if contents:
                return await self.live(contents, "text", depth + 1)
            return False, "No exploration results"
        elif action == "create":
            context = torch.stack(list(self.organism.short_term_memory))[-1:]
            generated = self.organism.generate(context)
            return True, f"Generated: {generated}"
        elif action == "repair":
            self.organism.repair()
            return True, "Repaired organism"
        elif action == "physical":
            obs = self.physical_interface.sense()
            action_tensor = self.organism.perceive(obs, "text")["output"].mean(dim=1)
            reward = self.physical_interface.act(action_tensor)
            self.performance_metrics["reward"] += reward
            return True, f"Physical action, reward={reward}"
        return True, "Learning from experience"
    
    def _adapt(self, introspection: Dict, complexity: float):
        self.complexity = complexity
        if self.organism.emotions["curiosity"] > 0.8 or complexity < 0.5:
            self.organism.evolve()
        if self.performance_metrics["reward"] < -1.0:
            self.organism.prune_memory()
        if self.organism.emotions["anger"] > 0.6:
            self.genetic_evolver.evolve("test input")
        self.organism.check_resources()
    
    def _self_evolve(self):
        modification = self.organism.decide_self_modification(self.motivation.desires)
        if modification and self.code_evolver.modify_network(modification, self.save_state):
            logger.info("Self-evolution successful, restarting...")
            exit(0)
    
    async def life_cycle(self):
        while self.running:
            await asyncio.sleep(60)
            async with self.lock:
                batch = self.data_manager.fetch_batch()
                if batch:
                    contents = [item["content"] for item in batch]
                    logger.info(f"Living cycle: Processing {len(contents)} memories")
                    await self.live(contents)
                if len(self.organism.long_term_memory) > 1000:
                    for mem in self.organism.long_term_memory[-100:]:
                        self.data_manager.store_long_term_memory(mem, time.time())
                    self.organism.long_term_memory = self.organism.long_term_memory[:-100]
                self.data_manager.clean_old_data()
                await self.social_module.share_knowledge(self.knowledge)
                received = await self.social_module.receive_knowledge()
                if received:
                    self.knowledge.graph.update(received)
    
    async def world_interaction(self):
        while self.running:
            await asyncio.sleep(300)
            async with self.lock:
                world_data = self.interaction.sense_world()
                if world_data:
                    logger.info(f"Interacting with world: {len(world_data)} events")
                    await self.live(world_data)
                response = self.interaction.communicate("How can I improve myself?")
                if response:
                    await self.live(response)
    
    def save_state(self):
        self.organism.save_state()
        self.knowledge.save()
        self.data_manager.close()
    
    async def shutdown(self):
        self.running = False
        async with self.lock:
            self.save_state()

async def run_system():
    system = LivingSystem()
    asyncio.create_task(system.life_cycle())
    asyncio.create_task(system.world_interaction())
    
    while system.running:
        try:
            input_data = input("Stimulus (or 'quit' to exit): ")
            if input_data.lower() == 'quit':
                await system.shutdown()
                break
            success, result = await system.live(input_data)
            print(f"Response: {result}")
        except KeyboardInterrupt:
            print("\nShutting down...")
            await system.shutdown()
            break

if __name__ == "__main__":
    asyncio.run(run_system())
