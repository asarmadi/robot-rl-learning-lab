import torch

class ReplayBuffer():
    def __init__(self, state_dim, size):
        self.state       = torch.zeros((size, state_dim))
        self.next_state  = torch.zeros((size, state_dim))
        self.reward      = torch.zeros((size, 1))
        self.action      = torch.zeros((size, 1))
        self.terminal    = torch.zeros((size, 1))


        self.idx = 0
        self.running_idx = 0
        self.size = size

    def __len__(self):
        return self.running_idx

    def getitem(self, batch_size):
        # Return one sample
        indices = torch.randint(0, self.running_idx, (batch_size,))
        return self.state[indices], self.next_state[indices], self.reward[indices], self.action[indices], self.terminal[indices]

    def additem(self, state, next_state, reward, action, d=0):
        self.state[self.idx]      = state.squeeze(-1)
        self.next_state[self.idx] = next_state.squeeze(-1)
        self.reward[self.idx]     = reward
        self.action[self.idx]     = action
        self.terminal[self.idx]   = d

        self.idx += 1
        
        if self.idx >= self.size:
            self.idx = 0
        else:
            self.running_idx += 1
