import torch
import torch.nn as nn
from typing import Dict, List
from config.settings import settings
from utils.logger import setup_logger
import copy
import numpy as np
from core.organism import Organism

logger = setup_logger(__name__)

class GeneticEvolver:
    def __init__(self, organism):
        self.organism = organism
        self.population = [self._mutate(organism.state_dict()) for _ in range(settings.GENETIC_POPULATION)]
    
    def _mutate(self, state_dict):
        new_state = {}
        for k, v in state_dict.items():
            if "weight" in k:
                new_state[k] = v + torch.randn_like(v) * 0.01
            else:
                new_state[k] = v
        organism = Organism()
        organism.load_state_dict(new_state)
        return organism
    
    def evaluate(self, inputs, modality="text") -> List[float]:
        fitness = []
        for individual in self.population:
            result = individual.perceive(inputs, modality)
            if result:
                fitness.append(result["awareness"].mean().item())
            else:
                fitness.append(0.0)
        return fitness
    
    def evolve(self, inputs, modality="text"):
        fitness = self.evaluate(inputs, modality)
        top_indices = np.argsort(fitness)[-2:]
        top_state_dicts = [self.population[i].state_dict() for i in top_indices]
        new_population = [self._create_individual(sd) for sd in top_state_dicts]
        
        while len(new_population) < settings.GENETIC_POPULATION:
            parent1, parent2 = random.sample(top_state_dicts, 2)
            child_state = self._crossover(parent1, parent2)
            new_population.append(self._mutate(child_state))
        
        self.population = new_population
        best = self.population[np.argmax(fitness)]
        self.organism.load_state_dict(best.state_dict())
        logger.info(f"Genetic evolution completed, best fitness={max(fitness)}")
    
    def _crossover(self, parent1, parent2):
        child = {}
        for k in parent1:
            if random.random() < 0.5:
                child[k] = parent1[k].clone()
            else:
                child[k] = parent2[k].clone()
        return child
    
    def _create_individual(self, state_dict):
        organism = Organism()
        organism.load_state_dict(state_dict)
        return organism