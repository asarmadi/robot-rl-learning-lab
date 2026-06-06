class Environment:
    def __init__(self):
        self.init_state = None
        self.current_state = self.init_state

    def reset(self):
        self.current_state = self.init_state

    def step(self, action):
        pass