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
        if self.type == 'discrete':
            logits       = self.predict(state)
            distribution = torch.distributions.Categorical(logits=logits)
            action       = distribution.sample()
            log_prob     = distribution.log_prob(action)
            action_env   = self.actions[action]
        else:
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

        return action, action_env, log_prob, distribution.entropy()