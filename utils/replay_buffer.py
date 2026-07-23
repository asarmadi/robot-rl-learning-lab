import torch

class ReplayBuffer():
    def __init__(self, state_dim, action_dim, size):
        self.state       = torch.zeros((size, state_dim))
        self.next_state  = torch.zeros((size, state_dim))
        self.reward      = torch.zeros((size, 1))
        self.action      = torch.zeros((size, action_dim))
        self.terminal    = torch.zeros((size, 1))


        self.idx = 0
        self.buffer_full = False # It checks when the buffer has all the elements filled
        self.size = size

    def __len__(self):
        if self.buffer_full:
            return self.size
        return self.idx

    def getitem(self, batch_size):
        # Return one sample
        if self.buffer_full:
            indices = torch.randint(0, self.size, (batch_size,))
        else:
            indices = torch.randint(0, self.idx, (batch_size,))
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
            self.buffer_full = True # The first time the buffer is full, it will be true for the rest of the training
