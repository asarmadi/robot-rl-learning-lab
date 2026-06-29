import torch
import torch.nn as nn
import torch.optim as optim

class Training:
    def __init__(self, batch_size, lr, gamma, policy, device='cpu', method='REINFORCE'):
        self.policy         = policy
        self.batch_size     = batch_size
        self.method         = method
        self.lr             = lr
        self.gamma          = gamma
        self.device         = device

    def train(self, states, actions, returns):

        optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.policy.train()
        
        optimizer.zero_grad()

        target = (reward.squeeze(-1) + values*self.gamma)

    
        output = self.online_network(state).gather(dim=1, index=action.long())
        loss = criterion(output.squeeze(-1), target)

        loss.backward()
        optimizer.step()

   