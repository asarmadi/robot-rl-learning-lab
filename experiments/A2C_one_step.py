import torch
import torch.nn as nn
import torch.optim as optim
from envs.cart_pole import CartPole
from estimators.mlp import MLP as MLP_V

# General hyper-parameters
n_episodes = 10000
gamma      = 0.995  # Discount factor
device     = 'cpu'
seed       = 42

# Policy hyper-parameters
action_type = 'continuous'
n_layers   = 2
hidden_dim = 128
action_dim = 2
actor_lr   = 0.0001
critic_lr = 0.0001
max_action = 5

torch.manual_seed(seed)


env = CartPole(method='A2C_one_step')
if action_type == 'discrete':
    from agents.mlp import MLP
    agent = MLP(input_dim=env.state_dim, output_dim=action_dim, n_layers=n_layers, hidden_dim=hidden_dim, max_action=max_action)
else:
    from agents.mlp_continuous import MLPC
    agent = MLPC(input_dim=env.state_dim, output_dim=action_dim, n_layers=n_layers, hidden_dim=hidden_dim, max_action=max_action)

V_phi = MLP_V(input_dim=env.state_dim,n_layers=n_layers, hidden_dim=hidden_dim, output_dim=1)

actor_optimizer = optim.Adam(agent.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(V_phi.parameters(), lr=critic_lr)
criterion   = nn.MSELoss()

rewards = []

for episode in range(1,n_episodes):
    env.reset()
    state = env.current_state
    print(f'Episode: {episode}')
    step = 0
    sum_rewards = 0

    while True:
        _, action_env, log_prob, _ = agent.get_action_g(state)

        next_state, reward, terminate = env.step(state, action_env, step)

        # TD Error:
        V_s = V_phi(state.squeeze(-1))
        if terminate == 'terminal':
            delta_t = reward - V_s.detach()
            target  = reward.unsqueeze(-1)
        else:
            V_s_1 = V_phi(next_state.squeeze(-1)).detach()
            delta_t = reward + gamma * V_s_1 - V_s
            target  = reward + gamma * V_s_1

        # Updating the critic
        critic_optimizer.zero_grad()
        critic_loss = criterion(V_s, target.detach())
        critic_loss.backward()
        critic_optimizer.step()

        # Updating the actor
        actor_optimizer.zero_grad()
        actor_loss = -(log_prob * delta_t.detach())
        actor_loss.backward()
        actor_optimizer.step()

        if terminate == 'terminal' or terminate == 'truncate':
            break
            
        state = next_state
        step += 1
        sum_rewards += reward
    print(f'Steps: {step}')
    rewards.append(sum_rewards)
    # plotting the result every 1000 episodes
    if episode % 200 == 0:
        env.test_policy(agent, episode//200)
        env.plot_rewards(rewards)