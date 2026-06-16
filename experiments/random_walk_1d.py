from agents.random_policy import RandomPolicy1D
from envs.random_walk import RandomWalk1D
from estimators.mc_value_estimator import MonteCarloValueEstimator

seed = 0
n_state = 7

policy = RandomPolicy1D(action_dim=2, seed=seed)
env    = RandomWalk1D(n_state=n_state)
mc     = MonteCarloValueEstimator(n_states=n_state, dir_='random_walk_1D/', gamma=0.9)

n_episodes = 10

for episode_idx in range(n_episodes):
    state = env.reset()
    episode = []
    while True:
        action = policy.act(state)
        next_state, reward = env.step(action)
        episode.append((state, reward))
        state = next_state
        if state == env.far_right_state or state == env.far_left_state:
            break
    
    mc.update(episode)

    if episode_idx % 1 == 0:
        mc.plot_states(episode_idx)
        mc.plot_value(episode_idx)