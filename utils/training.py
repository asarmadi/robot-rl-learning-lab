import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Training:
    def __init__(self, n_epochs, batch_size, lr, gamma, online_network, target_network, device='cpu', method='DQN'):
        self.online_network = online_network
        self.target_network = target_network
        self.n_epochs       = n_epochs
        self.batch_size     = batch_size
        self.method         = method
        self.lr             = lr
        self.gamma          = gamma
        self.device         = device

    def train(self, replay_buffer, copy_network=1):
        # copy_network is being used to copy the online weights to the target network
        # if copy_network is 0, we copy the network
        if copy_network == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        dataLoader = DataLoader(
            replay_buffer,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.online_network.parameters(), lr=self.lr)

        self.online_network.train()
        self.target_network.eval()
        for epoch in range(self.n_epochs):
            training_loss = 0
            for state, next_state, reward, action in dataLoader:
                optimizer.zero_grad()
                if self.method == 'DQN':
                   values, _ = torch.max(self.target_network.predict(next_state).squeeze(0),dim=1)
                   
                   target = (reward + values*self.gamma).unsqueeze(1)
                
                state = torch.as_tensor(state.squeeze(-1),dtype=torch.float32,device=self.device)
                output = self.online_network(state).gather(dim=1, index=action.unsqueeze(1))
                loss = criterion(output, target)
                training_loss += loss.item()

                loss.backward()
                optimizer.step()
            #print(f'Epoch: {epoch}, loss: {training_loss/len(dataLoader)}')

   