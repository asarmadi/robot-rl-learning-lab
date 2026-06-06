from envs.environment import Environment

class RandomWalk1D(Environment):
    def __init__(self, state_dim=1, action_dim=2, max_state=10):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.state_dim = self.state_dim
        self.init_state      = 0.0  
        self.far_right_state = max_state 
        self.far_left_state  = -max_state

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