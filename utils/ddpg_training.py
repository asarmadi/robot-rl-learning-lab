import torch
import torch.nn as nn
import torch.optim as optim

class Training:
    def __init__(self, Q_old, Q_new, actor_old, actor_new, config):
        self.Q_old     = Q_old
        self.Q_new     = Q_new
        self.actor_old = actor_old
        self.actor_new = actor_new
        self.config    = config
        self.criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.Q_new.parameters(),     lr=self.config['lr_critic'])
        self.actor_optimizer  = optim.Adam(self.actor_new.parameters(), lr=self.config['lr_actor'])

        # To make sure for the first trainig we start from what we used for data collection
        self.Q_old.load_state_dict(self.Q_new.state_dict())
        self.actor_old.load_state_dict(self.actor_new.state_dict())

    def train(self, replay_buffer):

        # Critic Update
        self.Q_new.train()
        self.actor_new.train()
        
        self.actor_old.eval()
        self.Q_old.eval()

        state, next_state, reward, action, d = replay_buffer.getitem(self.config['batch_size'])
        
        self.critic_optimizer.zero_grad()
        action_pred = self.actor_old.predict(next_state)
        input_ = torch.hstack((next_state,action_pred))
        values = self.Q_old.predict(input_)

        target = (reward + values*(1-d)*self.config['gamma'])
        input_new = torch.hstack((state,action))
        output    = self.Q_new(input_new)
        loss_acritic      = self.criterion(output, target.detach())

        loss_acritic.backward()
        self.critic_optimizer.step()
         
        # Actor Update

        # We want to update the actor and we are not updaring the critic at this stage
        # The gradient of the Q is wrt action not Q's own parameters
        for parameter in self.Q_new.parameters():
            parameter.requires_grad_(False)
            self.actor_optimizer.zero_grad()

        action_i = self.actor_new(state)
        input_i  = torch.hstack((state,action_i))
        loss_actor     = -self.Q_new(input_i).mean()

        loss_actor.backward()
        self.actor_optimizer.step()

        # To make sure, Q_new is available for updates in the next batch
        for parameter in self.Q_new.parameters():
            parameter.requires_grad_(True)
            self.actor_optimizer.zero_grad()

        # Soft update
        ## Action Value Update
        with torch.no_grad():
            for new_param, old_param in zip(self.Q_new.parameters(),self.Q_old.parameters()):
                old_param.mul_(1.0 - self.config['tau'])
                old_param.add_(self.config['tau'] * new_param)

        ## Policy Update
        with torch.no_grad():
            for new_param, old_param in zip(self.actor_new.parameters(),self.actor_old.parameters()):
                old_param.mul_(1.0 - self.config['tau'])
                old_param.add_(self.config['tau'] * new_param)
   