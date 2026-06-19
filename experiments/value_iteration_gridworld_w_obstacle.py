import numpy as np
from envs.gridworld import GridWorldwithObstacle
from agents.random_policy import RandomPolicy2D
from estimators.bellman import BellmanGridWorld2D


seed      = 0
grid_size = 4
gamma     = 0.9
save_rate = 1

env     = GridWorldwithObstacle(grid_size=grid_size)
agent   = RandomPolicy2D(action_dim=4, seed=seed) # Only defined to get the set of actions
plotter = BellmanGridWorld2D(grid_size=grid_size, dir_name='value_iteration_gridworld')
state_values = np.ones((grid_size,grid_size)) * -0.5
state_values[env.terminal_state[0], env.terminal_state[1]] = 0 # We want to keep the terminal state 
state_values[env.obstacle[0], env.obstacle[1]] = -10 # Since we are not updating the obstacle state, this value is not important.
#This is just for the plotting purposes 

delta = np.inf # The difference between current state_values and the new one
threshold = 0.01
iteration = 0  # For plotting purpose

while delta > threshold:
    delta = 0
    env.reset()
    state = env.current_state
    # Looping through all the states
    for row in range(grid_size):
        for col in range(grid_size):
            state = (row, col)
             # For the final state and the obstacle, we do not update. It should remain 0
            if (state == env.terminal_state).all() or (state == env.obstacle).all():
                continue
            current_v = state_values[row, col]
            q_a = []
            for action in agent.actions:
                prob, next_state, reward = env.update(state, action)
                row_n, col_n = next_state
                # We do not have the inner sigma since the transition to the next state given the action is deterministic
                q_a.append(prob * (reward + gamma * state_values[row_n, col_n]))

            updated_v = np.max(q_a)
            state_values[row, col] = updated_v
            delta = max(delta, np.abs(current_v-updated_v))

    if iteration % save_rate == 0:
        plotter.plot(state_values, iteration//save_rate)
    iteration += 1

