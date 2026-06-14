import numpy as np

class Agent:
    def __init__(self, action_dim, seed):
        self.action_dim = action_dim
        np.random.seed(seed)


    def act(self, state):
        action = np.random.randint(0, self.action_dim)
        return self.actions[action]