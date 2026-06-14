import numpy as np
from envs.environment import Environment

class GridWorld(Environment):
    def __init__(self, grid_size=4):
        super().__init__()
        self.init_state     = np.array([grid_size-1, 0])
        self.terminal_state = np.array([0,grid_size-1])
        self.grid_size = grid_size

    def update(self, state, action):
        # This function return probability of the next state, next_state and the reward
        next_state = tuple(state + action)

        ### We need to check whether the agent goes out of boundaries
        if next_state[0] < 0 or next_state[0] >= self.grid_size or \
            next_state[1] < 0 or next_state[1] >= self.grid_size:
            return 0, state, -1
        
        if (next_state == self.terminal_state).all():
            return 1, next_state, 0
        
        return 1, next_state, -1
