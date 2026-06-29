import torch

class ReplayBuffer():
    def __init__(self, state_dim, size):
        self.state       = torch.zeros((size, state_dim))
        self.next_state  = torch.zeros((size, state_dim))
        self.reward      = torch.zeros((size, 1))
        self.action      = torch.zeros((size, 1))

        self.idx = 0
        self.size = size

    def getitem(self, batch_size):
        # Return one sample
        indices = torch.randint(0, self.size, (batch_size,))
        return self.state[indices], self.next_state[indices], self.reward[indices], self.action[indices]

    def additem(self, state, next_state, reward, action):
        self.state[self.idx]      = state.squeeze(-1)
        self.next_state[self.idx] = next_state.squeeze(-1)
        self.reward[self.idx]     = reward
        self.action[self.idx]     = action

        self.idx += 1

        if self.idx >= self.size:
            self.idx = 0
