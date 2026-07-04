import torch
import torch.nn as nn
import torch.optim as optim
from envs.cart_pole import CartPole
from agents.mlp import MLP
from estimators.mlp import MLP as MLP_V

# General hyper-parameters
n_episodes = 10000
gamma      = 0.99  # Discount factor
device     = 'cpu'
step_size  = 5  # This is the n value for the future n steps of the algorithm
c_ent      = 0.001 # This is the entropy coefficient 

# Policy hyper-parameters
n_layers   = 2
hidden_dim = 128
action_dim = 2
actor_lr   = 0.0001
critic_lr = 0.0001
max_action = 2

env = CartPole(method='A2C_one_step')
agent = MLP(input_dim=env.state_dim,n_layers=n_layers, hidden_dim=hidden_dim, output_dim=action_dim, max_action=max_action)
V_phi = MLP_V(input_dim=env.state_dim,n_layers=n_layers, hidden_dim=hidden_dim, output_dim=1)

actor_optimizer = optim.Adam(agent.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(V_phi.parameters(), lr=critic_lr)
criterion   = nn.MSELoss()



for episode in range(1,n_episodes):
    env.reset()
    state = env.current_state
    print(f'Episode: {episode}')
    step_counter = 0
    states      = []
    actions     = []
    rewards     = []
    next_states = []
    terminates  = []

    while True:
        logits = agent(state.squeeze(-1))
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        next_state, reward, terminate = env.step(state, agent.actions[action])
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)

        if step_counter == step_size:
            step_counter = 0

            g = V_phi(states[i].squeeze(-1)).detach()
            i = len(rewards) - 1
            for reward in rewards[::-1]:
                # We use the same list to calculate the returns
                g = gamma * g + reward
                rewards[i] = g  # This actually has the targets for the critic
                i -= 1
        # TD Error:
        
        if terminate == 'terminal':
            delta_t = reward - V_s.detach()
            target  = torch.tensor(reward).to(device)
        else:
            V_s_1 = V_phi(next_state.squeeze(-1)).detach()
            delta_t = reward + gamma * V_s_1 - V_s
            target  = reward + gamma * V_s_1

        # Updating the critic
        critic_optimizer.zero_grad()
        critic_loss = criterion(V_s, torch.tensor(rewards))
        critic_loss.backward()
        critic_optimizer.step()

        # Updating the actor
        actor_optimizer.zero_grad()
        entropy_loss = distribution.entropy()
        actor_loss = -(log_prob * delta_t.detach()) - c_ent * entropy_loss
        actor_loss.backward()
        actor_optimizer.step()

        if terminate == 'terminal' or terminate == 'truncate':
            break
            
        state = next_state

    # plotting the result every 1000 episodes
    if episode % 50 == 0:
        env.test_policy(agent, episode//50)