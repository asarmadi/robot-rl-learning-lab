from envs.environment import Environment

class RandomWalk1D(Environment):
    def __init__(self):
        super().__init__(state_dim=1, action_dim=2)
        self.state_dim = self.state_dim
        self.init_state      = 0.0  
        self.far_right_state = 3.0  
        self.far_left_state  = -3.0

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