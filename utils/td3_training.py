import torch
import torch.nn as nn
import torch.optim as optim

class Training:
    def __init__(self, Q_old1, Q_old2, Q_new1, Q_new2, actor_old, actor_new, config):
        self.Q_old1     = Q_old1
        self.Q_old2     = Q_old2
        self.Q_new1     = Q_new1
        self.Q_new2     = Q_new2
        self.actor_old = actor_old
        self.actor_new = actor_new
        self.config    = config
        self.criterion = nn.MSELoss()
        self.critic_optimizer1 = optim.Adam(self.Q_new1.parameters(),     lr=self.config['lr_critic'])
        self.critic_optimizer2 = optim.Adam(self.Q_new2.parameters(),     lr=self.config['lr_critic'])
        self.actor_optimizer   = optim.Adam(self.actor_new.parameters(), lr=self.config['lr_actor'])

        # To make sure for the first trainig we start from what we used for data collection
        self.Q_old1.load_state_dict(self.Q_new1.state_dict())
        self.Q_old2.load_state_dict(self.Q_new2.state_dict())
        self.actor_old.load_state_dict(self.actor_new.state_dict())

    def train(self, replay_buffer, train_actor=1):
        # If train_actor is 0, we train the actor. This is part of delayed training of the TD3

        # Critic Update
        self.Q_new1.train()
        self.Q_new2.train()
        self.actor_new.train()
        
        self.Q_old1.eval()
        self.Q_old2.eval()
        self.actor_old.eval()


        state, next_state, reward, action, d = replay_buffer.getitem(self.config['batch_size'])
        
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        output = self.actor_old.predict(next_state)
        distribution    = torch.distributions.Normal(torch.zeros_like(output), 2)
        noise           = distribution.sample()
        action_pred     = output + noise

        input_  = torch.hstack((next_state,action_pred))
        values1 = self.Q_old1.predict(input_)
        values2 = self.Q_old2.predict(input_)
        values  = torch.min(values1, values2)

        target       = (reward + values*(1-d)*self.config['gamma'])
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

        # In TD3, only the first action value function is being used for the update
        if train_actor == 0:
            for parameter in self.Q_new1.parameters():
                parameter.requires_grad_(False)
                self.actor_optimizer.zero_grad()

            action_i = self.actor_new(state)
            input_i  = torch.hstack((state,action_i))
            loss_actor     = -self.Q_new1(input_i).mean()

            loss_actor.backward()
            self.actor_optimizer.step()

            # To make sure, Q_new is available for updates in the next batch
            for parameter in self.Q_new1.parameters():
                parameter.requires_grad_(True)
                self.actor_optimizer.zero_grad()

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

            ## Policy Update
            with torch.no_grad():
                for new_param, old_param in zip(self.actor_new.parameters(),self.actor_old.parameters()):
                    old_param.mul_(1.0 - self.config['tau'])
                    old_param.add_(self.config['tau'] * new_param)
   