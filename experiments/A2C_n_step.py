import torch
import torch.nn as nn

import torch.optim as optim
from envs.cart_pole import CartPole
from estimators.mlp import MLP as MLP_V


# General hyper-parameters
n_episodes = 10000
gamma      = 0.99  # Discount factor
device     = 'cpu'
step_size  = 5  # This is the n value for the future n steps of the algorithm
c_ent      = 0.01 # This is the entropy coefficient 
seed       = 42
# Policy hyper-parameters
action_type = 'continuous'
n_layers   = 2
hidden_dim = 128
action_dim = 2
actor_lr   = 0.001
critic_lr = 0.001
max_action = 2
torch.manual_seed(seed)

env = CartPole(method='A2C_n_step')
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


episode_rewards = []
for episode in range(1,n_episodes):
    env.reset()
    state = env.current_state
    print(f'Episode: {episode}')
    states      = torch.zeros((step_size, env.state_dim))
    actions     = torch.zeros((step_size, 1))
    rewards     = torch.zeros((step_size, 1))
    next_states = torch.zeros((step_size, env.state_dim))
    terminates  = []
    log_probs   = torch.zeros((step_size, 1))
    entropy_loss = torch.zeros((step_size, 1))
    sum_rewards = 0
    step        = 0
    buffer_idx  = 0

    while True:
        action, action_env, log_prob, entropy_l = agent.get_action_g(state)
        next_state, reward, terminate = env.step(state, action_env, step)
        
        states[buffer_idx]         = state
        actions[buffer_idx]        = action
        log_probs[buffer_idx]      = log_prob
        entropy_loss[buffer_idx]   = entropy_l
        rewards[buffer_idx]        = reward
        next_states[buffer_idx]    = next_state
        terminates.append(terminate)

        if buffer_idx == (step_size-1) or (buffer_idx < step_size and terminate == 'terminal') or (buffer_idx < step_size and terminate == 'truncate'):
            if terminates[-1] == 'terminal':
                g = 0
            else:
                g = V_phi(next_states[-1]).detach()
            
            returns = torch.zeros((buffer_idx+1,1)) # We use step as the size because in the latter part, the whole batch may not be full
            for idx in range(buffer_idx,-1,-1):
                # We use the same list to calculate the returns
                g = gamma * g + rewards[idx]
                returns[idx] = g  # This actually has the targets for the critic

            V_s = V_phi(states[:buffer_idx+1,:])
            # Updating the critic
            critic_optimizer.zero_grad()
            
            critic_loss = criterion(V_s, returns)
            critic_loss.backward()
            critic_optimizer.step()

            # Updating the actor
            actor_optimizer.zero_grad()
            actor_loss = -((log_probs[:buffer_idx+1,:] * returns - V_s.detach()).squeeze(-1)).mean() - c_ent * entropy_loss[:buffer_idx+1,:].mean()
            actor_loss.backward()
            actor_optimizer.step()

            states      = torch.zeros((step_size, env.state_dim))
            actions     = torch.zeros((step_size, 1))
            rewards     = torch.zeros((step_size, 1))
            next_states = torch.zeros((step_size, env.state_dim))
            terminates  = []
            log_probs   = torch.zeros((step_size, 1))
            entropy_loss = torch.zeros((step_size, 1))
            buffer_idx = 0


        if terminate == 'terminal' or terminate == 'truncate':
            break
            
        state = next_state
        step += 1
        buffer_idx += 1
        sum_rewards += reward
    episode_rewards.append(sum_rewards)

    # plotting the result every 1000 episodes
    if episode % 200 == 0:
        env.test_policy(agent, episode//200)
        env.plot_rewards(episode_rewards)