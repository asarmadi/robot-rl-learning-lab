import torch
from envs.cart_pole import CartPole
from estimators.mlp import MLP as MLP_V
from utils.ppo_training import Training
from utils.rollout_buffer import RolloutBuffer


# General hyper-parameters
method      = 'PPO'
action_type = 'discrete'
seed        = 42
n_episodes  = 10000
rollout_buffer_size  = 128  # This is different from the n-step, this is the size of the rollout buffer for training
gamma       = 0.99 # Discount factor
lambda_     = 0.95  # GAE weighting coefficient

# Agent hyper-parameters
n_layers   = 2
hidden_dim = 128
action_dim = 2
max_action = 2

# Training hyper-parameters
config = {
'lr' : 0.001,
'batch_size' : 32,
'n_epochs' : 20,
'epsilon'  : 0.2, # This is the clipping threshold for the actor loss
'c_ent'    : 0.0001 # Entropy loss coefficient
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
rollout_buffer = RolloutBuffer(state_dim=env.state_dim, size=rollout_buffer_size, gamma=gamma, lambda_=lambda_)

rewards = []
for episode in range(n_episodes):
    env.reset()
    state = env.current_state
    print(f'Episode: {episode}')
    entropy_loss = []
    step = 0
    sum_rewards = 0

    while True:
        if agent.type == 'discrete':
            logits = agent.predict(state.squeeze(-1))
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            action_env = agent.actions[action]
        else:
            action_mean, action_std = agent.predict(state.squeeze(-1))
            std = torch.exp(action_std) # To make sure the std is always positive
            distribution = torch.distributions.Normal(action_mean, std)
            action = distribution.sample()

            # Based on the mean and std, the action could be out of the desired range, we need to correct that
            squashed_action = torch.tanh(action)       
            action_env = action_max * squashed_action          

            log_prob = distribution.log_prob(raw_action)

            log_prob -= torch.log(1.0 - squashed_action.pow(2) + 1e-6)

            log_prob -= torch.log(torch.as_tensor(action_max, device=raw_action.device))

        entropy_loss.append(distribution.entropy())
        next_state, reward, terminate = env.step(state, action_env, step)
        rollout_buffer.additem(state.squeeze(-1),next_state, reward, action, log_prob, terminate)

        if len(rollout_buffer) == rollout_buffer_size:
            rollout_buffer.cal_advantages(V_phi)
            training.train(rollout_buffer)
            rollout_buffer.reset()
            entropy_loss = []

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