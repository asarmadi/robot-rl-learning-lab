import torch
from envs.cart_pole import CartPole
from agents.mlp import MLP
from estimators.mlp import MLP as MLP_V
from utils.reinforce_training import Training
from utils.value_training import ValueTraining

method = 'REINFORCE_w_baseline'
# Genral hyper-parameters
n_episodes = 4001
gamma      = 0.9 # Discount factor
action_dim = 10
seed       = 0

# Policy hyper-parameters
hidden_dim = 128
n_layers   = 2
max_action = 2

# Policy training hyper-parameters
lr = 0.001
device = 'cpu'

torch.manual_seed(seed)

env   = CartPole(method=method)
agent = MLP(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=action_dim, max_action=max_action)
V_phi = MLP_V(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=1) # This is the value function estimator, the output is one as we estimate the expected return for a given state 
training  = Training(lr, agent, device=device, method=method, value_function=V_phi)
v_training = ValueTraining(lr, value_network=V_phi, device=device)

for episode in range(1,n_episodes):
    print(f'Episode: {episode}')
    env.reset()
    state = env.current_state
    states  = []
    actions = []
    rewards = []

    while True:
        # Here the cart pole has a discrete output, therefore, we are choosing the maximum as the action
        # For each bin, we have corresponding probabilities
        logits = agent(state.squeeze(-1))
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        breakpoint()

        next_state, reward, terminate = env.step(state, agent.actions[action])

        if terminate:
            break

        state = next_state
        states.append(state.squeeze(-1))
        actions.append(action.unsqueeze(-1))
        rewards.append(reward)
    
    g = 0
    i = len(rewards) - 1
    for reward in rewards[::-1]:
        # We use the same list to calculate the returns
        g = gamma * g + reward
        rewards[i] = g
        i -= 1

    v_training.train(torch.stack(states), torch.tensor(rewards))

    training.train(torch.stack(states), torch.stack(actions), torch.tensor(rewards))

    # plotting the result every 1000 episodes
    if episode % 1000 == 0:
        env.test_policy(agent, episode)
