import numpy as np

class Agent:
    def __init__(self, action_dim, seed, type_):
        self.action_dim = action_dim
        self.type = type_
        np.random.seed(seed)


    def act(self, state):
        action = np.random.randint(0, self.action_dim)
        return self.actions[action]

    
    def get_sample(self, state):
        if self.type == 'discrete':
            logits = self.predict(state)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            action_env = agent.actions[action]
        else:
            action_mean, action_std = agent.predict(state)
            std = torch.exp(action_std) # To make sure the std is always positive
            distribution = torch.distributions.Normal(action_mean, std)
            action = distribution.sample()

            # Based on the mean and std, the action could be out of the desired range, we need to correct that
            squashed_action = torch.tanh(action)       
            action_env = action_max * squashed_action          

            log_prob = distribution.log_prob(raw_action)

            log_prob -= torch.log(1.0 - squashed_action.pow(2) + 1e-6)

            log_prob -= torch.log(torch.as_tensor(action_max, device=raw_action.device))

        return action, action_env, log_prob