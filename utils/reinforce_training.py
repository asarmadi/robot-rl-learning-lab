import torch
import torch.nn as nn
import torch.optim as optim

class Training:
    def __init__(self, lr, policy, device='cpu', method='REINFORCE', value_function=None):
        self.policy         = policy
        self.method         = method
        self.lr             = lr
        self.device         = device
        if method == 'REINFORCE_w_baseline':
            self.value_function = value_function

    def train(self, states, actions, returns):

        optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.policy.train()
        if self.method == 'REINFORCE_w_baseline':
            self.value_function.eval()
        
        optimizer.zero_grad()
        
        #loss = -(torch.log(self.policy(states).gather(dim=1, index=actions.long())).squeeze(-1)*returns).sum() This Gives me nan
        if self.method == 'REINFORCE':
            loss = -(torch.log_softmax(self.policy(states), dim=1).gather(dim=1, index=actions.long()).squeeze(-1) * returns).mean()
        elif self.method == 'REINFORCE_w_baseline':
            V_phi = self.value_function(states).squeeze(-1)
            loss = -(torch.log_softmax(self.policy(states), dim=1).gather(dim=1, index=actions.long()).squeeze(-1) * (returns - V_phi.detach())).mean()


        loss.backward()
        optimizer.step()

   