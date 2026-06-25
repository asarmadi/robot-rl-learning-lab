import torch
import torch.nn as nn
import torch.optim as optim

class Training:
    def __init__(self, n_epochs, batch_size, lr, online_network, target_network, method='DQN'):
        self.online_network = online_network
        self.target_network = target_network
        self.n_epochs       = n_epochs
        self.batch_size     = batch_size
        self.method         = method
        self.lr             = lr

def train(replay_buffer, copy_network=1):
    # copy_network is being used to copy the online weights to the target network
    # if copy_network is 0, we copy the network
    if copy_network == 0:
        self.target_network.load_state_dict(self.online_model.state_dict())

    dataLoader = DataLoader(
        replay_buffer,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=2
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.online_network.parameters(), lr=self.lr)

    self.online_network.train()
    self.target_network.eval()
    for epoch in range(self.n_epochs):
        for state, next_state, reward, action in dataLoader:
            optimizer.zero_grad()
            if self.method == 'DQN':
               target = reward + self.gamma*max(self.target_network(state))
            output = self.online_network(state)[action]
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

   