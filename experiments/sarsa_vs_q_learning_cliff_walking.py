import os
import numpy as np
from envs.cliff_walking import cliffWalking
from agents.epsilon_greedy import epsilonGreedy
import matplotlib.pyplot as plt

seed       = 42
n_episodes = 500
grid_size  = (4, 12)
action_dim = 4
alpha      = 0.1
gamma      = 0.5
epsilon    = 0.1
methods    = ['sarsa', 'q_learning']

env       = cliffWalking(grid_size=grid_size)
agent     = epsilonGreedy(action_dim=action_dim, seed=seed, epsilon=epsilon)

sum_rewards_dict = {}
## This part presents the Sarsa Control
for method in methods:
    Q_sarsa = np.zeros((grid_size[0], grid_size[1], action_dim))
    sum_rewards_dict[method] = []
    for episode_idx in range(n_episodes):
        sum_rewards = 0
        env.reset()
        state = env.current_state
        row, col = state[0], state[1]
        action = agent.act(Q_sarsa[row, col, :])

        while True:
            row, col = state[0], state[1]
            next_state, reward = env.update(state, agent.actions[action])
            row_n, col_n = next_state[0], next_state[1]
            action_n = agent.act(Q_sarsa[row_n, col_n, :])

            if method == 'sarsa':
                Q_sarsa[row, col, action] = Q_sarsa[row, col, action] + alpha * (reward + gamma * Q_sarsa[row_n, col_n, action_n] - Q_sarsa[row, col, action])
            elif method == 'q_learning':
                Q_sarsa[row, col, action] = Q_sarsa[row, col, action] + alpha * (reward + gamma * np.max(Q_sarsa[row_n, col_n, :]) - Q_sarsa[row, col, action])

            sum_rewards += reward
            if (next_state == env.terminal_state).all():
                break

            state = next_state
            action = action_n
        sum_rewards_dict[method].append(sum_rewards)

save_dir = './logs/sarsa_vs_qlearning_cliffWalking/'
os.makedirs(save_dir, exist_ok=True)
plt.figure(0)

for key, value in sum_rewards_dict.items():
    plt.plot(value, label=key)

plt.xlabel("Episodes")
plt.ylabel("Sum of rewards during episode")
plt.legend()

plt.savefig(f'{save_dir}compare.png')
plt.close()



