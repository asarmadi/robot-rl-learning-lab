import torch
from agents.mlp_continuous import MLPC
from envs.cart_pole import CartPole
from utils.replay_buffer import ReplayBuffer
from estimators.mlp import MLP as MLPQ
from utils.ddpg_training import Training

# General hyper-parameters
seed        = 42
method      = 'DDPG'
buffer_size = 100000
n_episodes  = 10000
initial_n_samples = 2000 # This is the number of samples we collect before we start training

torch.manual_seed(seed)
# Agent/Policy hyper-parameters
action_type = 'deterministicContinuous' # This will output the action directly, instead of a probability distribution
hidden_dim  = 128
n_layers    = 2
output_dim  = 1
max_action  = 5

# Training hyper_parameters
config = {
    'lr_actor': 0.0001,
    'lr_critic': 0.001,
    'gamma': 0.99, # Discount factor
    'tau':   0.005, # Soft update constant
    'batch_size': 128,
}

env   = CartPole(method=method)
agent_new = MLPC(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=output_dim, type_=action_type, max_action=max_action)
agent_old = MLPC(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=output_dim, type_=action_type, max_action=max_action)

Q_new     = MLPQ(input_dim=env.state_dim+output_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=1)
Q_old     = MLPQ(input_dim=env.state_dim+output_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=1)

replay_buffer = ReplayBuffer(size=buffer_size, state_dim=env.state_dim)
training      = Training(Q_old=Q_old, Q_new=Q_new, actor_old=agent_old, actor_new=agent_new, config=config)

rewards     = []

for episode in range(1, n_episodes):
    env.reset()
    state = env.current_state
    sum_rewards = 0
    step        = 0

    while True:
        action, action_env, _ = agent_new.get_action(state)
        next_state, reward, terminate = env.step(state, action_env, step)

        d = 0 # This is going to be used for calculating the target value
        if terminate == 'terminal':
            d = 1
        replay_buffer.additem(state, next_state, reward, action, d=d)

        if len(replay_buffer) >= initial_n_samples:
            training.train(replay_buffer)

        if terminate == 'terminal' or terminate == 'truncate':
            break

        step += 1
        sum_rewards += reward
        state = next_state

    rewards.append(sum_rewards)
    print(f'Episode: {episode}, Steps: {step}, Reward: {sum_rewards}')

    # plotting the result every 1000 episodes
    if episode % 200 == 0:
        env.test_policy(agent_new, episode//200)
        env.plot_rewards(rewards)