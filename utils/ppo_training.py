import torch
import torch.nn as nn
import torch.optim as optim

class Training:
    def __init__(self,policy, state_value, config):
        self.policy                = policy
        self.state_value           = state_value
        self.config                = config
        self.state_value_criterion = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.policy.parameters(), lr=config['lr'])
        self.critic_optimizer = optim.Adam(self.state_value.parameters(), lr=config['lr'])

    def train(self, rollout_buffer):
        # Training the actor
        self.policy.train()
        self.state_value.eval()

        for epoch in range(self.config['n_epochs']):
            state, advantage, prev_policy = rollout_buffer.getitem(self.config['batch_size'])
        
            self.actor_optimizer.zero_grad()
            r_t = self.policy(state)/prev_policy
            loss = torch.clip(r_t, min=(1-self.config['epsilon']), max=(1+self.config['epsilon']))*advantage

            loss.backward()
            self.actor_optimizer.step()
        
        # Training the Critic
        for epoch in range(self.config['n_epochs']):
            state, advantage, _ = rollout_buffer.getitem(self.config['batch_size'])
            self.critic_optimizer.zero_grad()
            output = self.state_value(state)
            loss = self.state_value_criterion(advantage+output.detach(), output)

            loss.backward()
            self.critic_optimizer.step()

   