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
step_size  = 20  # This is the n value for the future n steps of the algorithm
c_ent      = 0.01 # This is the entropy coefficient 

# Policy hyper-parameters
n_layers   = 2
hidden_dim = 128
action_dim = 2
actor_lr   = 0.0001
critic_lr = 0.001
max_action = 2

env = CartPole(method='A2C_n_step')
agent = MLP(input_dim=env.state_dim,n_layers=n_layers, hidden_dim=hidden_dim, output_dim=action_dim, max_action=max_action)
V_phi = MLP_V(input_dim=env.state_dim,n_layers=n_layers, hidden_dim=hidden_dim, output_dim=1)

actor_optimizer = optim.Adam(agent.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(V_phi.parameters(), lr=critic_lr)
criterion   = nn.MSELoss()



for episode in range(1,n_episodes):
    env.reset()
    state = env.current_state
    print(f'Episode: {episode}')
    states      = []
    actions     = []
    rewards     = []
    next_states = []
    terminates  = []
    log_probs   = []
    entropy_loss = []

    while True:
        logits = agent(state.squeeze(-1))
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_probs.append(distribution.log_prob(action))
        entropy_loss.append(distribution.entropy())
        next_state, reward, terminate = env.step(state, agent.actions[action])
        states.append(state.squeeze(-1))
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        terminates.append(terminate)

        if len(states) == step_size or (len(states) < step_size and terminate == 'terminal') or (len(states) < step_size and terminate == 'truncate'):

            if terminates[-1] == 'terminal':
                g = 0
            else:
                g = V_phi(next_states[-1].squeeze(-1)).detach()
            i = len(rewards) - 1
            for reward in rewards[::-1]:
                # We use the same list to calculate the returns
                g = gamma * g + reward
                rewards[i] = g  # This actually has the targets for the critic
                i -= 1

            V_s = V_phi(torch.stack(states))
            # Updating the critic
            critic_optimizer.zero_grad()
            critic_loss = criterion(V_s, torch.stack(rewards))
            critic_loss.backward()
            critic_optimizer.step()

            # Updating the actor
            actor_optimizer.zero_grad()
            actor_loss = -((torch.stack(log_probs) * (torch.stack(rewards) - V_s.detach()).squeeze(-1))).mean() - c_ent * torch.stack(entropy_loss).mean()
            actor_loss.backward()
            actor_optimizer.step()

            states          = []
            actions         = []
            rewards         = []
            next_states     = []
            terminates      = []
            log_probs       = []
            entropy_loss    = []

        if terminate == 'terminal' or terminate == 'truncate':
            break
            
        state = next_state

    # plotting the result every 1000 episodes
    if episode % 50 == 0:
        env.test_policy(agent, episode//50)