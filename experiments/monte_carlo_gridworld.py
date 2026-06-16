from envs.gridworld import GridWorld
from agents.random_policy import RandomPolicy2D
from estimators.mc_value_estimator import MonteCarloValueEstimator

seed = 0
n_episodes = 10000
save_rate  = 1000  # To make sure, we are saving every 1000 steps
grid_size  = 4
n_actions  = 4
gamma      = 1

env   = GridWorld(grid_size=grid_size, mode='mc')
agent = RandomPolicy2D(action_dim=n_actions, seed=seed)
estimator = MonteCarloValueEstimator(gamma=gamma, n_states=n_actions, dir_='bellman_gridworld/') # n_states is not needed for this problem, I just filled it as a placeholder



for episode_idx in range(n_episodes):
    env.reset()
    state = env.current_state
    episode = []
    while True:
        action     = agent.act(state)
        next_state, reward = env.update(state=state, action=action)
        episode.append((tuple(state), reward))
        state =  next_state
        if (state == env.terminal_state).all():
            episode.append((tuple(state), reward))
            break
    estimator.update(episode)

    if episode_idx % save_rate == 0:
        estimator.plot_states_gridworld(dir_= episode_idx // save_rate, grid_size=grid_size)