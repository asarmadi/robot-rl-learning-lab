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
step_size  = 128  # This is different from the n-step, this is the size of the rollout buffer for training
gamma      = 0.99 # Discount factor
lambda_    = 0.9  # GAE weighting coefficient

# Agent hyper-parameters
n_layers   = 2
hidden_dim = 128
action_dim = 2
max_action = 2

# Training hyper-parameters
config = {
'lr' : 0.001,
'batch_size' : 128,
'n_epochs' : 100
}

torch.manual_seed(seed)

env   = CartPole(method=method)
agent = MLP(input_dim=env.state_dim, output_dim=action_dim, n_layers=n_layers, hidden_dim=hidden_dim, max_action=max_action)
V_phi = MLP_V(input_dim=env.state_dim, n_layers=n_layers, hidden_dim=hidden_dim, output_dim=1)

training       = Training(policy=agent, state_value=V_phi, config=config)
rollout_buffer = RolloutBuffer(state_dim=env.state_dim, size=step_size, gamma=gamma, lambda_=lambda_)


for episode in range(n_episodes):
    env.reset()
    state = env.current_state
    print(f'Episode: {episode}')
    entropy_loss = []

    while True:
        logits = agent(state.squeeze(-1))
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_probs = distribution.log_prob(action)
        entropy_loss.append(distribution.entropy())
        next_state, reward, terminate = env.step(state, agent.actions[action])
        rollout_buffer.additem(state.squeeze(-1),next_state, reward, action, log_probs)

        if len(entropy_loss) == step_size or (len(entropy_loss) < step_size and terminate == 'terminal') or (len(entropy_loss) < step_size and terminate == 'truncate'):
            rollout_buffer.cal_advantages()
            training.train(rollout_buffer)
            rollout_buffer.reset()
            entropy_loss = []

        if terminate == 'terminal' or terminate == 'truncate':
            break
            
        state = next_state

    # plotting the result every 1000 episodes
    if episode % 50 == 0:
        env.test_policy(agent, episode//50)