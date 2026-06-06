import numpy as np
from agents.agent import Agent

class RandomPolicy(Agent):
    def __init__(self, action_dim, seed):
        super().__init__(action_dim)
        self.actions = [-1, 1]  # Example actions for a random walk
        np.random.seed(seed)

    def act(self, state, key=None):
        action = np.random.randint(0, self.action_dim)
        return self.actions[action]