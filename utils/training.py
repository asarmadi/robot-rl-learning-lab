import torch
import torch.nn as nn
import torch.optim as optim

class Training:
    def __init__(self, batch_size, lr, gamma, online_network, target_network, device='cpu', method='DQN'):
        self.online_network = online_network
        self.target_network = target_network
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

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.online_network.parameters(), lr=self.lr)

        self.online_network.train()
        self.target_network.eval()

        state, next_state, reward, action = replay_buffer.getitem(self.batch_size)
        
        optimizer.zero_grad()
        if self.method == 'DQN':
           values, _ = torch.max(self.target_network.predict(next_state),dim=1)
           
        elif self.method == 'DDQN':
           _, indices = torch.max(self.online_network.predict(next_state),dim=1)
           values = self.target_network.predict(next_state)[:,indices]

        target = (reward.squeeze(-1) + values*self.gamma)

    
        output = self.online_network(state).gather(dim=1, index=action.long())
        loss = criterion(output.squeeze(-1), target)

        loss.backward()
        optimizer.step()
            #print(f'Epoch: {epoch}, loss: {training_loss/len(dataLoader)}')

   