from envs.cart_pole import CartPole
from agents.mlp import MLP

# Genral hyper-parameters
n_episodes = 10000
gamma      = 0.9 # Discount factor

# Policy hyper-parameters
hidden_dim = 128
n_layers   = 2


env   = CartPole(method='REINFORCE')
agent = MLP(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=env.action_dim)

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
        rewards[i] = gamma * g + reward
        i -= 1
