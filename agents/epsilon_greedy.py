import numpy as np
from agents.agent import Agent

class epsilonGreedy(Agent):
    def __init__(self, action_dim, seed, epsilon):
        super().__init__(seed=seed, action_dim=action_dim)
        self.epsilon = epsilon
        self.actions = [np.array([-1,0]), np.array([1,0]), np.array([0,1]), np.array([0,-1])] # Down, Up, Right, Left
        

    def act(self, action_value):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        max_indices = np.argmax(action_value)
        if type(max_indices) == list:
            idx = np.random.randint(len(max_indices))
            return max_indices[idx]
        return max_indices

    def act_greedy(self, action_value):
        max_indices = np.argmax(action_value)
        if type(max_indices) == list:
            idx = np.random.randint(len(max_indices))
            return max_indices[idx]
        return max_indices

