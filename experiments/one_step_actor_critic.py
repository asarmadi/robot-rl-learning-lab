import torch.optim as optim
from envs.cart_pole import CartPole
from agents.mlp import MLP
from estimators.mlp import MLP as MLP_V

# General hyper-parameters
n_episodes = 10000

# Policy hyper-parameters
n_layers   = 2
hidden_dim = 128
action_dim = 2
actor_lr   = 0.001
critic_lr = 0.001

env = CartPole()
agent = MLP(n_layers=n_layers, hidden_dim=hidden_dim, output_dim=action_dim)
V_phi = MLP_V(n_layers=n_layers, hidden_dim=hidden_dim, output_dim=1)

actor_optimizer = optim.Adam(agent.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(V_phi.parameters(), lr=critic_lr)


for episode_idx in range(n_episodes):
    env.reset()
    state = env.current_state

    while True:
        logits = agent.predict(state.squeeze(-1)).detach()
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        next_state, reward, terminate = env.step(state, agent.actions[action])

        # TD Error:
        if terminate == 'terminal':
            delta_t = reward - V_phi(state)
        else:
            delta_t = reward + gamma * V_phi(next_state) - V_phi(state)

        # Updating the critic
        critic_optimizer.zero_grad()
        criterion = nn.MSELoss()
        critic_loss = criterion(V_phi(next_state), delta_t)
        critic_loss.backward()
        critic_optimizer.step()

        # Updating the actor
        actor_optimizer.zero_grad()
        actor_loss = -(torch.log_softmax(logits, dim=1).gather(dim=1, index=action.long()).squeeze(-1) * delta_t.detach())
        actor_loss.backward()
        actor_optimizer.step()

        if terminate == 'terminal' or terminate == 'truncate':
            break
            
        state = next_state