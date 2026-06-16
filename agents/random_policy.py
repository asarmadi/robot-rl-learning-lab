import numpy as np
from agents.agent import Agent

class RandomPolicy1D(Agent):
    def __init__(self, action_dim, seed):
        super().__init__(action_dim, seed)
        self.actions = [-1, 1]  # Example actions for a random walk

    def act(self, state):
        return super().act(state)
    
class RandomPolicy2D(Agent):
    def __init__(self, action_dim, seed):
        super().__init__(action_dim, seed)
        self.actions = [np.array([-1,0]), np.array([1,0]), np.array([0,1]), np.array([0,-1])] # Down, Up, Right, Left

    def act(self, state):
        return super().act(state)