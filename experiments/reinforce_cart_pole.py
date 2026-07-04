import torch
import numpy as np
from envs.cart_pole import CartPole
from agents.mlp import MLP
from utils.reinforce_training import Training

# Genral hyper-parameters
n_episodes = 10001
gamma      = 0.99 # Discount factor
action_dim = 20
seed       = 0
epsilon    = 0.9 # I added this to encourage exploration 


# Policy hyper-parameters
hidden_dim = 128
n_layers   = 2
max_action = 2

# Policy training hyper-parameters
lr = 0.001
device = 'cpu'

torch.manual_seed(seed)

env   = CartPole(method='REINFORCE')
agent = MLP(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=action_dim, max_action=max_action)
training  = Training(lr, agent, device='cpu', method='REINFORCE')

for episode in range(1,n_episodes):
    print(f'Episode: {episode}')
    env.reset()
    state = env.current_state
    states  = []
    actions = []
    rewards = []

    step = 0

    while True:
        # Here the cart pole has a discrete output, therefore, we are choosing the maximum as the action
        # For each bin, we have corresponding probabilities
        logits = agent(state.squeeze(-1))
        distribution = torch.distributions.Categorical(logits=logits.detach())
        action = distribution.sample()
        next_state, reward, terminate = env.step(state, agent.actions[action])

        if terminate == 'terminal' or terminate == 'truncate':
            break

        state = next_state
        states.append(state.squeeze(-1))
        actions.append(action.unsqueeze(-1))
        rewards.append(reward)
        
        step += 1

        if step > 1000:
            break

    g = 0
    i = len(rewards) - 1
    for reward in rewards[::-1]:
        # We use the same list to calculate the returns
        g = gamma * g + reward
        rewards[i] = g
        i -= 1
    
    training.train(torch.stack(states), torch.stack(actions), torch.tensor(rewards))

    # plotting the result every 1000 episodes
    if episode % 1000 == 0:
        env.test_policy(agnet, episode)
