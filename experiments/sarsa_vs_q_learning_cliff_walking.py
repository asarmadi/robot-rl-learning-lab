import os
import numpy as np
from envs.cliff_walking import cliffWalking
from agents.epsilon_greedy import epsilonGreedy
import matplotlib.pyplot as plt

seed       = 42
n_episodes = 500
grid_size  = (4, 12)
action_dim = 4
alpha      = 0.05
gamma      = 1
epsilon    = 0.1
max_action = 2
methods    = ['sarsa', 'q_learning']
save_dir = './logs/sarsa_vs_qlearning_cliffWalking/'

os.makedirs(save_dir, exist_ok=True)
os.makedirs(f'{save_dir}action_values/', exist_ok=True)

env       = cliffWalking(grid_size=grid_size)
agent     = epsilonGreedy(action_dim=action_dim, seed=seed, epsilon=epsilon, max_action=max_action)

sum_rewards_dict = {}
## This part presents the Sarsa Control
for method in methods:
    Q = np.zeros((grid_size[0], grid_size[1], action_dim))
    sum_rewards_dict[method] = []
    for episode_idx in range(n_episodes):
        sum_rewards = 0 # To keep the sum of the rewards in each episode
        n_steps     = 0 # To count number of steps for each episode
        env.reset()
        state = env.current_state.copy()
        row, col = state[0], state[1]
        action = agent.act(Q[row, col, :])
        while True:
            row, col = state[0], state[1]
            next_state, reward = env.update(state, agent.actions[action])
            row_n, col_n = next_state[0], next_state[1]
            action_n = agent.act(Q[row_n, col_n, :])
            if method == 'sarsa':
                Q[row, col, action] = Q[row, col, action] + alpha * (reward + gamma * Q[row_n, col_n, action_n] - Q[row, col, action])
            elif method == 'q_learning':
                Q[row, col, action] = Q[row, col, action] + alpha * (reward + gamma * np.max(Q[row_n, col_n, :]) - Q[row, col, action])

            sum_rewards += reward
            if (next_state == env.terminal_state).all():
                break

            state = next_state
            action = action_n
            n_steps += 1
        sum_rewards_dict[method].append(sum_rewards)

    fig, ax = plt.subplots()
    grid = np.zeros(grid_size)

    ## Following the optimal policy
    env.reset()
    state = env.current_state.copy()
    while not (state == env.terminal_state).all():
        row, col = state[0], state[1]
        grid[row,col] = 1.0
        action = agent.act_greedy(Q[row, col, :])
        state += agent.actions[action]


    ax.imshow(grid,
              origin="lower",
              extent=(0, grid_size[1], 0, grid_size[0])
              )

    # Display each value inside its cell
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            ax.text(
                col+0.5,
                row+0.5,
                f"{grid[row, col]:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title(f"Optimal Path via {method}")
    ax.set_xticks(np.arange(0, grid_size[1]+1, 1))
    ax.set_yticks(np.arange(0, grid_size[0]+1, 1))
    ax.grid()
    ax.tick_params()
    fig.savefig(f'{save_dir}action_values/{method}.png')
    plt.close(fig)
    plt.close()

plt.figure(0)

for key, value in sum_rewards_dict.items():
    plt.plot(value, label=key)

plt.xlabel("Episodes")
plt.ylabel("Sum of rewards during episode")
plt.legend()

plt.savefig(f'{save_dir}compare.png')
plt.close()



