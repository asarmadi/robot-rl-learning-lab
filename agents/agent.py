import torch
import numpy as np

class Agent:
    def __init__(self, action_dim, seed, type_, **kwargs):
        self.action_dim = action_dim
        self.type = type_
        np.random.seed(seed)
        super().__init__(**kwargs)


    def act(self, state):
        action = np.random.randint(0, self.action_dim)
        return self.actions[action]

    
    def get_action(self, state):
        log_prob = None # By default, it is none
        if self.type == 'discrete':
            logits       = self.predict(state)
            distribution = torch.distributions.Categorical(logits=logits)
            action       = distribution.sample()
            log_prob     = distribution.log_prob(action)
            action_env   = self.actions[action]
        elif self.type == 'continuous':
            output = self.predict(state)
            if state.ndim == 1:
                out_mean, out_std = output[0], output[1]
            else:
                out_mean, out_std = output[:,0:1], output[:,1:2]
            std          = torch.exp(out_std) # To make sure the std is always positive
            distribution = torch.distributions.Normal(out_mean, std)
            action       = distribution.sample()

            # Based on the mean and std, the action could be out of the desired range, we need to correct that
            squashed_action = torch.tanh(action)       
            action_env = self.max_action * squashed_action          

            log_prob = distribution.log_prob(action)

            log_prob -= torch.log(1.0 - squashed_action.pow(2) + 1e-6)

            log_prob -= torch.log(torch.as_tensor(self.max_action, device=action.device))

        elif self.type == 'directContinuous':
            output = self.predict(state)
            squashed_action = torch.tanh(output)
            action = self.max_action * squashed_action
            action_env = action # Since the action is exactly the output of the network

        return action, action_env, log_prob

    def get_action_g(self, state, action=None):
        action_env = None
        log_probs = None
        dist_entropy = None
        if self.type == 'discrete':
            logits = self.forward(state)
            distribution = torch.distributions.Categorical(logits=logits)
            # This is for the case where we are doing a rollout
            if action == None:
                action       = distribution.sample()
                log_probs     = distribution.log_prob(action)
                action_env   = self.actions[action]
            else:
                # This is for the case where we want to use the sample to train, therefore, we use the action collected during rollout
                log_probs = distribution.log_prob(action).unsqueeze(-1)

            dist_entropy = distribution.entropy()

        elif self.type == 'continuous':
            output = self.forward(state)
            if state.ndim == 1:
                # During the rollout we only pass one state at a time
                out_mean, out_std = output[0], output[1]
            else:
                # Druing training we pass a batch of samples
                out_mean, out_std = output[:,0:1], output[:,1:2]

            out_std = torch.clamp(out_std, self.min_log_std, self.max_log_std)
            std = torch.exp(out_std)

            distribution = torch.distributions.Normal(out_mean, std)
            if action == None:
                action       = distribution.sample()
                # Based on the mean and std, the action could be out of the desired range, we need to correct that
                squashed_action = torch.tanh(action)       
                action_env = self.max_action * squashed_action

            squashed_action = torch.tanh(action)       

            log_probs = distribution.log_prob(action)

            log_probs -= torch.log(1.0 - squashed_action.pow(2) + 1e-6)

            log_probs -= torch.log(torch.as_tensor(self.max_action, device=action.device))

            dist_entropy = distribution.entropy()
        elif self.type == 'directContinuous':
            output = self.forward(state)
            squashed_action = torch.tanh(output)
            action = self.max_action * squashed_action
            action_env = action
              
        return action, action_env, log_probs, dist_entropy
        