import numpy as np
from envs.gridworld import GridWorld
from agents.random_policy import RandomPolicy2D
from estimators.bellman import BellmanGridWorld2D

grid_size  = 4
n_episodes = 1000
gamma      = 1
state_values = np.zeros((grid_size, grid_size))
state_values[0, grid_size-1] = 0 # This is the terminal state and should remain 0

seed = np.random.seed(42)

env = GridWorld(grid_size=grid_size)
agent = RandomPolicy2D(action_dim=4, seed=seed)
estimator = BellmanGridWorld2D() # This is just used for plotting purposes

iteration  = 0
delta_     = np.inf     # The error between future and current estimates of value function
threshold  = 0.1 # This is used as the stop criteria

while delta_ > threshold:
    delta_ = 0
    for row in range(grid_size):
        for col in range(grid_size):
            state = (row, col)
            # For the final state, we do not update. It should remain 0
            if state == (0, grid_size-1):
                continue
            value_ = 0

            for action_ in agent.actions:
                # Since the agent is using a random policy, the probability of taking each action
                # equals to 1/4
                prob, next_state, reward = env.update(state, action_)
                # The reason we do not see the summation because it is deterministic
                value_ += ((1/len(agent.actions)) * prob * (reward + gamma * state_values[next_state]))
            delta_ = max(delta_, abs(value_-state_values[state]))
            state_values[state] = value_
    estimator.plot(state_values, iteration)
    iteration += 1
