import torch
from envs.cart_pole import CartPole
from estimators.mlp import MLP as MLP_V
from utils.ppo_training import Training
from utils.rollout_buffer import RolloutBuffer


# General hyper-parameters
method      = 'PPO'
action_type = 'continuous'
seed        = 42
n_episodes  = 10000
rollout_buffer_size  = 128  # This is different from the n-step, this is the size of the rollout buffer for training
gamma       = 0.995 # Discount factor
lambda_     = 0.95  # GAE weighting coefficient

# Agent hyper-parameters
n_layers   = 2
hidden_dim = 128
if action_type == 'discrete':
    action_dim = 2
else:
    action_dim = 1
max_action = 5

# Training hyper-parameters
config = {
'lr' : 0.0001,
'batch_size' : 32,
'n_epochs' : 5,
'epsilon'  : 0.2, # This is the clipping threshold for the actor loss
'c_ent'    : 0.001 # Entropy loss coefficient
}

torch.manual_seed(seed)

env   = CartPole(method=method)
if action_type == 'discrete':
    from agents.mlp import MLP
    agent = MLP(input_dim=env.state_dim, output_dim=action_dim, n_layers=n_layers, hidden_dim=hidden_dim, max_action=max_action)
else:
    from agents.mlp_continuous import MLPC
    agent = MLPC(input_dim=env.state_dim, output_dim=action_dim, n_layers=n_layers, hidden_dim=hidden_dim, max_action=max_action)

V_phi = MLP_V(input_dim=env.state_dim, n_layers=n_layers, hidden_dim=hidden_dim, output_dim=1)

training       = Training(policy=agent, state_value=V_phi, config=config)
rollout_buffer = RolloutBuffer(state_dim=env.state_dim, size=rollout_buffer_size, gamma=gamma, lambda_=lambda_, batch_size=config['batch_size'])

rewards = []
for episode in range(n_episodes):
    env.reset()
    state = env.current_state
    print(f'Episode: {episode}')
    step = 0
    sum_rewards = 0

    while True:
        action, action_env, log_prob = agent.get_action(state)

        next_state, reward, terminate = env.step(state, action_env, step)
        rollout_buffer.additem(state,next_state, reward, action, log_prob, terminate)

        if len(rollout_buffer) == rollout_buffer_size:
            rollout_buffer.cal_advantages(V_phi)
            training.train(rollout_buffer)
            rollout_buffer.reset()

        if terminate == 'terminal' or terminate == 'truncate':
            break
            
        state = next_state
        step += 1
        sum_rewards += reward
    rewards.append(sum_rewards)
    print(f'Steps: {step}, Reward= {sum_rewards}')

    # plotting the result every 1000 episodes
    if episode % 200 == 0:
        env.test_policy(agent, episode//200)
        env.plot_rewards(rewards)