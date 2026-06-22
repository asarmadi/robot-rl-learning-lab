import numpy as np
from envs.environment import Environment

class cliffWalking(Environment):
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.init_state = np.array([0,0])
        self.terminal_state = np.array([0, grid_size[1]-1])

    def update(self, state, action):
        next_state = state + action
        # Out of boundary states
        if next_state[0] < 0 or next_state[0] >= self.grid_size[0] or \
           next_state[1] < 0 or next_state[1] >= self.grid_size[1]:
           return state, -1

        # Terminal state
        if (next_state == self.terminal_state).all():
            return next_state, 0

        # Hitting the cliff
        if next_state[0] == self.init_state[0] and next_state[1] > self.init_state[1] and next_state[1] < self.terminal_state[1]:
            return self.init_state, -100

        # All other cases
        return next_state, -1

