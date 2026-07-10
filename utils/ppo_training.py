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
            state, advantage, prev_policy, action, _ = rollout_buffer.getitem(self.config['batch_size'])
        
            self.actor_optimizer.zero_grad()
            logits = self.policy(state)
            if torch.isnan(logits).any():
                breakpoint()
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(action).unsqueeze(-1)
            entropy_loss = distribution.entropy()
            r_t = torch.exp(log_probs - prev_policy.detach())
            actor_loss = -torch.min(r_t*advantage.detach(),(torch.clip(r_t, min=(1-self.config['epsilon']), max=(1+self.config['epsilon']))*advantage.detach())).mean() - self.config['c_ent'] * entropy_loss.mean()
            actor_loss.backward()
            self.actor_optimizer.step()
        
        # Training the Critic
        for epoch in range(self.config['n_epochs']):
            state, advantage, _, _, value = rollout_buffer.getitem(self.config['batch_size'])
            self.critic_optimizer.zero_grad()
            output = self.state_value(state)
            critic_loss = self.state_value_criterion(advantage.detach()+value, output)

            critic_loss.backward()
            self.critic_optimizer.step()

   