import torch

class RolloutBuffer():
    def __init__(self, state_dim, size, gamma, lambda_):
        self.size      = size
        self.state_dim = state_dim
        self.gamma     = gamma
        self.lambda_   = lambda_
        self.reset()

    def getitem(self, batch_size):
        # Return one sample
        indices = torch.randint(0, self.size, (batch_size,))
        return self.state[indices], self.advantages[indices], self.log_prob[indices], self.action[indices], self.values[indices]

    def additem(self, state, next_state, reward, action, log_prob, terminate):
        self.state[self.idx]        = state.squeeze(-1)
        self.next_state[self.idx]   = next_state.squeeze(-1)
        self.reward[self.idx]       = reward
        self.action[self.idx]       = action
        self.log_prob[self.idx]     = log_prob
        self.terminate[self.idx]    = terminate
        self.idx += 1

    def cal_advantages(self, V_phi):
        self.advantages = torch.zeros((self.size+1, 1))
        self.values = torch.zeros((self.size+1, 1))

        # We already defined the A[t+1] = 0
        for t in range(self.size-1,-1,-1):
            self.values[t] = V_phi(self.state[t]).detach()
            if self.terminate[t] == 'terminal':
                delta_t = self.reward[t] - self.values[t]
            else:
                delta_t = self.reward[t] + self.gamma * V_phi(self.next_state[t]).detach() - self.values[t]
            self.advantages[t] = delta_t + self.gamma * self.lambda_ *  self.advantages[t+1]

        # We normalize the advantage to get a less noisier training loss
        mean = self.advantages.mean(dim=0)
        std  = self.advantages.std(dim=0, unbiased=False)
        
        self.advantages = (self.advantages - mean) / (std + 1e-8)

    def __len__(self):
        return self.idx

    def reset(self):
        self.state         = torch.zeros((self.size, self.state_dim))
        self.next_state    = torch.zeros((self.size, self.state_dim))
        self.reward        = torch.zeros((self.size, 1))
        self.action        = torch.zeros((self.size, 1))
        self.terminate     = {}
        self.log_prob      = torch.zeros((self.size, 1))
        self.advantages    = torch.zeros((self.size, 1))

        self.idx = 0

