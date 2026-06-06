from envs.environment import Environment

class RandomWalk1D(Environment):
    def __init__(self, n_state=10):
        super().__init__()
        self.init_state      = 0
        self.far_right_state = n_state//2 + 1
        self.far_left_state  = -self.far_right_state - 1

    def reset(self):
        self.current_state = self.init_state
        return self.current_state

    def step(self, action):
        self.current_state += action
        if self.current_state == self.far_right_state:
            reward = 1.0
        else:
            reward = 0.0
        return self.current_state, reward