import torch
import torch.nn as nn
import torch.optim as optim

class Training:
    def __init__(self, Q_old1, Q_old2, Q_new1, Q_new2, actor, config):
        self.Q_old1            = Q_old1
        self.Q_old2            = Q_old2
        self.Q_new1            = Q_new1
        self.Q_new2            = Q_new2
        self.actor             = actor
        self.config            = config
        self.criterion         = nn.MSELoss()
        self.critic_optimizer1 = optim.Adam(self.Q_new1.parameters(), lr=self.config['lr_critic'])
        self.critic_optimizer2 = optim.Adam(self.Q_new2.parameters(), lr=self.config['lr_critic'])
        self.actor_optimizer   = optim.Adam(self.actor.parameters(),  lr=self.config['lr_actor'])

        # To make sure for the first trainig we start from what we used for data collection
        self.Q_old1.load_state_dict(self.Q_new1.state_dict())
        self.Q_old2.load_state_dict(self.Q_new2.state_dict())

        # We are not training the Q_old networks, we only use the soft update to update them from new networks
        for parameter in self.Q_old1.parameters():
            parameter.requires_grad_(False)

        for parameter in self.Q_old2.parameters():
            parameter.requires_grad_(False)

    def train(self, replay_buffer):
        # Critic Update
        self.Q_new1.train()
        self.Q_new2.train()
        self.actor.train()
        
        self.Q_old1.eval()
        self.Q_old2.eval()

        state, next_state, reward, action, d = replay_buffer.getitem(self.config['batch_size'])
        
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        _, action_pred, log_prob = self.actor.get_action(next_state)

        input_  = torch.hstack((next_state,action_pred))
        values1 = self.Q_old1.predict(input_)
        values2 = self.Q_old2.predict(input_)
        values  = torch.min(values1, values2)

        target       = (reward + (values-self.config['alpha']*log_prob)*(1-d)*self.config['gamma'])
        input_new    = torch.hstack((state,action))
        output1      = self.Q_new1(input_new)
        loss_critic1 = self.criterion(output1, target.detach())

        loss_critic1.backward()
        self.critic_optimizer1.step()

        output2      = self.Q_new2(input_new)
        loss_critic2 = self.criterion(output2, target.detach())

        loss_critic2.backward()
        self.critic_optimizer2.step()
         
        # Actor Update

        # We want to update the actor and we are not updaring the critic at this stage
        # The gradient of the Q is wrt action not Q's own parameters

        for parameter in self.Q_new1.parameters():
            parameter.requires_grad_(False)

        for parameter in self.Q_new2.parameters():
            parameter.requires_grad_(False)
        
        self.actor_optimizer.zero_grad()

        _, action_i, log_probs, _ = self.actor.get_action_g(state)
        input_i    = torch.hstack((state,action_i))
        values11   = self.Q_new1.predict(input_i)
        values12   = self.Q_new2.predict(input_i)
        values1    = torch.min(values11, values12)
        loss_actor = self.config['alpha']*log_probs.mean() - values1.mean()

        loss_actor.backward()
        self.actor_optimizer.step()

        # To make sure, Q_new is available for updates in the next batch
        for parameter in self.Q_new1.parameters():
            parameter.requires_grad_(True)

        for parameter in self.Q_new2.parameters():
            parameter.requires_grad_(True)

        # Soft update
        ## Action Value Update
        with torch.no_grad():
            for new_param, old_param in zip(self.Q_new1.parameters(),self.Q_old1.parameters()):
                old_param.mul_(1.0 - self.config['tau'])
                old_param.add_(self.config['tau'] * new_param)

        with torch.no_grad():
            for new_param, old_param in zip(self.Q_new2.parameters(),self.Q_old2.parameters()):
                old_param.mul_(1.0 - self.config['tau'])
                old_param.add_(self.config['tau'] * new_param)
   