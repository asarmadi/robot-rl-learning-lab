from agents.random_policy import RandomPolicy
from envs.random_walk import RandomWalk1D
from estimators.mc_value_estimator import MonteCarloValueEstimator


policy = RandomPolicy(action_dim=2)
env    = RandomWalk1D()
mc     = MonteCarloValueEstimator(env.state_dim, 'random_walk_1D/')

n_episodes = 100

for episode_idx in range(n_episodes):
    state = env.reset()
    done = False
    episode = []
    while not done:
        action = policy.act(state)
        next_state, reward = env.step(action)
        episode.append((state, reward))
        state = next_state
        if state == env.far_right_state or state == env.far_left_state:
            done = True
    
    mc.update(episode)

    if episode_idx % 10 == 0:
        mc.plot_states(episode_idx)
        mc.plot_value(episode_idx)