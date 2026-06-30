import torch
import torch.nn as nn
import torch.optim as optim

class Training:
    def __init__(self, lr, policy, device='cpu', method='REINFORCE'):
        self.policy         = policy
        self.method         = method
        self.lr             = lr
        self.device         = device

    def train(self, states, actions, returns):

        optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.policy.train()
        
        optimizer.zero_grad()

        loss = -(torch.log(self.policy(states).gather(dim=1, index=actions.long()))*returns).sum(dim=1)

        loss.backward()
        optimizer.step()

   