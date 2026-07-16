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
        self.critic_optimizer = optim.Adam(self.Q_new.parameters(),     lr=self.config['lr'])
        self.actor_optimizer  = optim.Adam(self.actor_new.parameters(), lr=self.config['lr'])

    def train(self, replay_buffer):

        # Critic Update
        self.Q_new.train()
        self.actor_old.eval()

        state, next_state, reward, action, d = replay_buffer.getitem(self.config['batch_size'])
        
        optimizer.zero_grad()
        action_pred = self.actor_old.predict(next_state)
        input_ = torch.hstack((next_state,action_pred))
        values = self.Q_old.predict(input_)

        target = (reward + values*(1-d)*self.config['gamma'])

        input_new = torch.hstack((state,action))
        output    = self.Q_new(input_new)
        loss      = self.criterion(output, target)

        loss.backward()
        critic_optimizer.step()
         
        # Actor Update
        action_i = self.actor_new(state)
        input_i  = torch.hstack((state,action_i))
        loss     = -self.Q_new(input_i).mean()

        loss.backward()
        actor_optimizer.step()

        # Soft update
        ## Action Value Update
        with torch.no_grad():
            for new_param, old_param in zip(Q_new.parameters(),Q_old.parameters()):
                old_param.mul_(1.0 - self.config['tau'])
                old_param.add_(self.config['tau'] * new_param)

        ## Policy Update
        with torch.no_grad():
            for new_param, old_param in zip(actor_new.parameters(),actor_old.parameters()):
                old_param.mul_(1.0 - self.config['tau'])
                old_param.add_(self.config['tau'] * new_param)
   