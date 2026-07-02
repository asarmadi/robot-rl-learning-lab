import torch.nn as nn
import torch.optim as optim

class ValueTraining:
    def __init__(self, lr, value_network, device='cpu'):
        self.value_network = value_network
        self.lr             = lr
        self.device         = device

    def train(self, states, returns):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr)

        self.value_network.train()
        
        optimizer.zero_grad()
    
        output = self.value_network(states)
        loss = criterion(output.squeeze(-1), returns)

        loss.backward()
        optimizer.step()

   