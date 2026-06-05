class Environment:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.init_state = None
        self.current_state = self.init_state

    def reset(self):
        self.current_state = self.init_state

    def step(self, action):
        pass