import torch

class RolloutBuffer():
    def __init__(self, state_dim, action_dim, size, gamma, lambda_, batch_size):
        self.size      = size
        self.state_dim = state_dim
        self.gamma     = gamma
        self.lambda_   = lambda_
        self.batch_size= batch_size 
        self.action_dim= action_dim
        self.reset()

    def getitem(self, idx):
        # Return one sample
        start = idx*self.batch_size
        end   = (idx+1)*self.batch_size
        mini_batch = self.indices[start:end]
        return self.state[mini_batch], self.advantages[mini_batch], self.log_prob[mini_batch], self.action[mini_batch], self.values[mini_batch]

    def additem(self, state, next_state, reward, action, log_prob, terminate):
        self.state[self.idx]        = state
        self.next_state[self.idx]   = next_state
        self.reward[self.idx]       = reward
        self.action[self.idx]       = action
        self.log_prob[self.idx]     = log_prob
        self.terminate[self.idx]    = terminate
        self.idx += 1

    def cal_advantages(self, V_phi):
        self.advantages = torch.zeros((self.size, 1))
        self.values = torch.zeros((self.size, 1))
        
        # We already defined the A[t+1] = 0
        for t in range(self.size-1,-1,-1):
            self.values[t] = V_phi(self.state[t]).detach()
            if self.terminate[t] == 'terminal':
                self.advantages[t] = self.reward[t] - self.values[t]
            else:
                delta_t = self.reward[t] + self.gamma * V_phi(self.next_state[t]).detach() - self.values[t]
                if t == (self.size-1):
                    self.advantages[t] = delta_t
                else:
                    self.advantages[t] = delta_t + self.gamma * self.lambda_ *  self.advantages[t+1]

        # We normalize the advantage to get a less noisier training loss
        self.mean = self.advantages.mean(dim=0)
        self.std  = self.advantages.std(dim=0, unbiased=False)
        
    def normalize(self, input_):
        return (input_ - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return self.idx

    def shuffle(self):
        self.indices =  torch.randint(0, self.size, (self.size,))

    def reset(self):
        self.state         = torch.zeros((self.size, self.state_dim))
        self.next_state    = torch.zeros((self.size, self.state_dim))
        self.reward        = torch.zeros((self.size, 1))
        self.action        = torch.zeros((self.size, self.action_dim))
        self.terminate     = {}
        self.log_prob      = torch.zeros((self.size, 1))
        self.advantages    = torch.zeros((self.size, 1))

        self.idx = 0

