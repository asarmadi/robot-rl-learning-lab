import numpy as np
from envs.environment import Environment

class GridWorld(Environment):
    def __init__(self, grid_size=4, mode='bellman'):
        super().__init__()
        self.init_state     = np.array([0, 0])
        self.terminal_state = np.array([grid_size-1, grid_size-1])
        self.grid_size = grid_size
        self.mode = mode

    def update(self, state, action):
        # This function return probability of the next state, next_state and the reward
        next_state = state + action

        ### We need to check whether the agent goes out of boundaries
        if next_state[0] < 0 or next_state[0] >= self.grid_size or \
            next_state[1] < 0 or next_state[1] >= self.grid_size:
            prob, next_state, reward=  1, state, -1
        
        elif (next_state == self.terminal_state).all():
            prob, next_state, reward = 1, next_state, 0
        else:
            prob, next_state, reward = 1, next_state, -1

        if self.mode == 'mc':
            return next_state, reward
        else:
            return prob, next_state, reward
