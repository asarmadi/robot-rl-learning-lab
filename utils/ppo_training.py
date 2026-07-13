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
            rollout_buffer.shuffle()
            for idx in range(len(rollout_buffer)//self.config['batch_size']):
                state, advantage, prev_policy, action, _ = rollout_buffer.getitem(idx)
                norm_adv = rollout_buffer.normalize(advantage)

                self.actor_optimizer.zero_grad()
                _, _, log_probs, entropy_loss = self.policy.get_action_g(state, action)
                r_t = torch.exp(log_probs - prev_policy.detach())
                actor_loss = -torch.min(r_t*norm_adv,(torch.clip(r_t, min=(1-self.config['epsilon']), max=(1+self.config['epsilon']))*norm_adv)).mean() - self.config['c_ent'] * entropy_loss.mean()
                actor_loss.backward()
                self.actor_optimizer.step()
        
    
                state, advantage, _, _, value = rollout_buffer.getitem(idx)
                self.critic_optimizer.zero_grad()
                output = self.state_value(state)
                critic_loss = self.state_value_criterion(output, advantage.detach()+value)

                critic_loss.backward()
                self.critic_optimizer.step()

   