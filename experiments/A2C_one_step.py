import torch
import torch.nn as nn
import torch.optim as optim
from envs.cart_pole import CartPole
from agents.mlp import MLP
from estimators.mlp import MLP as MLP_V

# General hyper-parameters
n_episodes = 10000
gamma      = 0.99  # Discount factor

# Policy hyper-parameters
n_layers   = 2
hidden_dim = 128
action_dim = 2
actor_lr   = 0.001
critic_lr = 0.001
max_action = 2

env = CartPole(method='A2C_one_step')
agent = MLP(input_dim=env.state_dim,n_layers=n_layers, hidden_dim=hidden_dim, output_dim=action_dim, max_action=max_action)
V_phi = MLP_V(input_dim=env.state_dim,n_layers=n_layers, hidden_dim=hidden_dim, output_dim=1)

actor_optimizer = optim.Adam(agent.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(V_phi.parameters(), lr=critic_lr)


for episode in range(1,n_episodes):
    env.reset()
    state = env.current_state
    print(f'Episode: {episode}')

    while True:
        logits = agent(state.squeeze(-1))
        distribution = torch.distributions.Categorical(logits=logits.detach())
        action = distribution.sample()
        next_state, reward, terminate = env.step(state, agent.actions[action])

        # TD Error:
        if terminate == 'terminal':
            delta_t = reward - V_phi(state)
            target  = reward
        else:
            delta_t = reward + gamma * V_phi(next_state.squeeze(-1)) - V_phi(state.squeeze(-1))
            target  = reward + gamma * V_phi(next_state.squeeze(-1))

        # Updating the critic
        critic_optimizer.zero_grad()
        criterion = nn.MSELoss()
        critic_loss = criterion(V_phi(next_state.squeeze(-1)), target.detach())
        critic_loss.backward()
        critic_optimizer.step()

        # Updating the actor
        actor_optimizer.zero_grad()
        actor_loss = -(torch.log_softmax(logits, dim=0).gather(dim=0, index=action.long()) * delta_t.detach())
        actor_loss.backward()
        actor_optimizer.step()

        if terminate == 'terminal' or terminate == 'truncate':
            break
            
        state = next_state

    # plotting the result every 1000 episodes
    if episode % 1000 == 0:
        env.test_policy(agent, episode)