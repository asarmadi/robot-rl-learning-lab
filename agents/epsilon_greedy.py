import numpy as np
from agents.agent import Agent

class epsilonGreedy(Agent):
    def __init__(self, action_dim, seed, epsilon_max, max_action=2, epsilon_min=0.05, decay_rate=0.00001, environment='GridWorld'):
        super().__init__(seed=seed, action_dim=action_dim)
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_rate  = decay_rate
        self.epsilon     = epsilon_max

        if environment == 'GridWorld':
            self.actions = [np.array([-1,0]), np.array([1,0]), np.array([0,1]), np.array([0,-1])] # Down, Up, Right, Left
        else:
            self.actions = np.linspace(-max_action,max_action,action_dim)

    def act(self, action_value):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        max_indices = np.argmax(action_value)
        if type(max_indices) == list:
            idx = np.random.randint(len(max_indices))
            return max_indices[idx]
        return max_indices

    def act_greedy(self, action_value):
        max_indices = np.argmax(action_value)
        if type(max_indices) == list:
            idx = np.random.randint(len(max_indices))
            return max_indices[idx]
        return max_indices
    
    def set_epsilon(self, step):
        '''
        I added this function to be able to control exploration
        '''
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.decay_rate * step)

