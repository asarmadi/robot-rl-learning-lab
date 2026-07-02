from envs.cart_pole import CartPole
from agents.mlp import MLP
from estimators.mlp import MLP as MLP_V
from utils.reinforce_training import Training

# Genral hyper-parameters
n_episodes = 10000
gamma      = 0.9 # Discount factor

# Policy hyper-parameters
hidden_dim = 128
n_layers   = 2

# Policy training hyper-parameters
lr = 0.001
device = 'cpu'


env            = CartPole(method='REINFORCE')
agent          = MLP(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=env.action_dim)
value_function = MLP_V(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=1)
training       = Training(lr, policy, device='cpu', method='REINFORCE')
training_       = Training(lr, policy, device='cpu', method='REINFORCE')


for episode in range(n_episodes):
    env.reset()
    state = env.current_state
    states  = []
    actions = []
    rewards = []

    while True:
        action = agent.predict(state.squeeze(-1)).detach().numpy()
        next_state, reward, terminate = env.step(state, action)

        if terminate:
            break

        state = next_state
        states.append(state)
        actions.append(actions)
        rewards.append(reward)

    g = 0
    i = len(rewards) - 1
    for reward in rewards[::-1]:
        # We use the same list to calculate the returns
        rewards[i] = gamma * g + reward
        i -= 1

    training.train(torch.stack(states), torch.stack(actions), torch.stack(rewards))
