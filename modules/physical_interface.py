import gym
from utils.logger import setup_logger
import torch

logger = setup_logger(__name__)

class PhysicalInterface:
    def __init__(self):
        try:
            self.env = gym.make("CartPole-v1")
            self.env.reset()
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            self.env = None
    
    def sense(self):
        if self.env is None:
            return torch.zeros(4, dtype=torch.float32)
        try:
            observation = self.env.step(self.env.action_space.sample())[0]
            return torch.tensor(observation, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Sense error: {e}")
            return torch.zeros(4, dtype=torch.float32)
    
    def act(self, action: torch.Tensor):
        if self.env is None:
            return 0.0
        try:
            action_idx = torch.argmax(action).item()
            obs, reward, done, _ = self.env.step(action_idx)
            if done:
                self.env.reset()
            return reward
        except Exception as e:
            logger.error(f"Act error: {e}")
            return 0.0