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
            if self.policy.type == 'discrete':
                logits = self.policy(state)
                distribution = torch.distributions.Categorical(logits=logits)
                log_probs = distribution.log_prob(action).unsqueeze(-1)
            else:
                mu, action_std = self.policy(state)
                std = torch.exp(action_std)
                distribution = torch.distributions.Normal(mu, std)
                squashed_action = torch.tanh(action)       
    
                log_probs = distribution.log_prob(action)

                log_probs -= torch.log(1.0 - squashed_action.pow(2) + 1e-6)

                log_probs -= torch.log(torch.as_tensor(self.config['action_max'], device=action.device))

            entropy_loss = distribution.entropy()
            r_t = torch.exp(log_probs - prev_policy.detach())
            actor_loss = -torch.min(r_t*advantage,(torch.clip(r_t, min=(1-self.config['epsilon']), max=(1+self.config['epsilon']))*advantage)).mean() - self.config['c_ent'] * entropy_loss.mean()
            actor_loss.backward()
            self.actor_optimizer.step()
        
        # Training the Critic
        for epoch in range(self.config['n_epochs']):
            state, advantage, _, _, value = rollout_buffer.getitem(self.config['batch_size'])
            self.critic_optimizer.zero_grad()
            output = self.state_value(state)
            critic_loss = self.state_value_criterion(output, advantage.detach()+value)

            critic_loss.backward()
            self.critic_optimizer.step()

   