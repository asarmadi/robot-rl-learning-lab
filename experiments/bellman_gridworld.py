import numpy as np
from envs.gridworld import GridWorld
from agents.random_policy import RandomPolicy2D
from estimators.bellman import BellmanGridWorld2D

grid_size  = 4
n_episodes = 1000
gamma      = 1
state_values = np.ones((grid_size, grid_size)) * 0.5
state_values[grid_size-1, grid_size-1] = 0

seed = 42

env = GridWorld(grid_size=grid_size, mode='bellman')
agent = RandomPolicy2D(action_dim=4, seed=seed)
estimator = BellmanGridWorld2D(grid_size=grid_size) # This is just used for plotting purposes

iteration  = 0
delta_     = np.inf     # The error between future and current estimates of value function
threshold  = 0.01 # This is used as the stop criteria
save_rate  = 100  # To limit the number of images to be saved

while delta_ > threshold:
    delta_ = 0
    for row in range(grid_size):
        for col in range(grid_size):
            state = (row, col)
            # For the final state, we do not update. It should remain 0
            if (state == env.terminal_state).all():
                continue
            value_ = 0
            pi_a_s = 1/len(agent.actions)

            for action_ in agent.actions:
                # Since the agent is using a random policy, the probability of taking each action
                # equals to 1/4
                prob, next_state, reward = env.update(state, action_)
                # The reason we do not see the summation because it is deterministic
                n_row, n_col = next_state
                value_ += (pi_a_s * prob * (reward + gamma * state_values[n_row, n_col]))
            delta_ = max(delta_, abs(value_-state_values[row, col]))
            state_values[row, col] = value_
    if iteration % save_rate == 0:
        estimator.plot(state_values, iteration // save_rate)
    iteration += 1
