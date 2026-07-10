import torch
from envs.cart_pole import CartPole
from agents.mlp import MLP
from estimators.mlp import MLP as MLP_V
from utils.ppo_training import Training
from utils.rollout_buffer import RolloutBuffer


# General hyper-parameters
method     = 'PPO'
seed       = 42
n_episodes = 10000
rollout_buffer_size  = 128  # This is different from the n-step, this is the size of the rollout buffer for training
gamma      = 0.99 # Discount factor
lambda_    = 0.95  # GAE weighting coefficient

# Agent hyper-parameters
n_layers   = 2
hidden_dim = 128
action_dim = 2
max_action = 2

# Training hyper-parameters
config = {
'lr' : 0.0001,
'batch_size' : 32,
'n_epochs' : 4,
'epsilon'  : 0.2, # This is the clipping threshold for the actor loss
'c_ent'    : 0.001 # Entropy loss coefficient
}

torch.manual_seed(seed)

env   = CartPole(method=method)
agent = MLP(input_dim=env.state_dim, output_dim=action_dim, n_layers=n_layers, hidden_dim=hidden_dim, max_action=max_action)
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
        logits = agent(state.squeeze(-1))
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_probs = distribution.log_prob(action)
        entropy_loss.append(distribution.entropy())
        next_state, reward, terminate = env.step(state, agent.actions[action], step)
        rollout_buffer.additem(state.squeeze(-1),next_state, reward, action, log_probs, terminate)

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