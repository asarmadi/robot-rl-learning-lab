import torch
from envs.cart_pole import CartPole
from agents.epsilon_greedy import epsilonGreedy
from estimators.mlp import MLP
from utils.training import Training
from utils.replay_buffer import ReplayBuffer

seed              = 0
torch.manual_seed(seed)

# DQN Hyper-Parameters
n_episodes         = 100
epsilon            = 0.7  # Probability for taking random actions
action_dim         = 10   # Number of bins for the action space
gamma              = 0.9  # Discount factor
initial_data_size  = 1000 # This is used to make sure, we train the network after dataset has some samples and not train only on few samples in the begining
update_rate        = 500 # Number of steps used to copy the online network to the target network.
replay_buffer_size = 100000

# Training Hyper-Parameters
batch_size        = 64
lr                = 0.001

# Q Neural Networks Hyper-Parameters
hidden_dim        = 128
n_layers          = 2
device            = 'cpu'





env      = CartPole(method='DQN')
agent    = epsilonGreedy(epsilon=epsilon, action_dim=action_dim, seed=seed, environment='CartPole')
Q_online = MLP(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=agent.action_dim)
Q_target = MLP(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=agent.action_dim)

training = Training(batch_size=batch_size,
                    lr=lr,
                    gamma=gamma,
                    device=device,
                    target_network=Q_target,
                    online_network=Q_online)

replay_buffer = ReplayBuffer(size=replay_buffer_size,state_dim=env.state_dim)

step = 0

for episode in range(n_episodes):
    print(f'Episode: {episode}')
    env.reset()
    state = env.current_state

    while True:
        state = torch.as_tensor(state,dtype=torch.float32,device=device)
        action = agent.act(Q_online.predict(state.squeeze(-1)).detach().numpy())
        next_state, reward, reached_goal = env.step(state, agent.actions[action])
        replay_buffer.additem(state,next_state,reward,action)

        if reached_goal:
            break

        step += 1
        state = next_state
        
        if (step > initial_data_size): # We update the online network after collecting inital number of samples
            training.train(replay_buffer, copy_network=(step % update_rate))


# Testing the optimal policy
states_for_plotting = []
env.reset()
state = env.current_state

while True:
    state = torch.as_tensor(state,dtype=torch.float32,device=device)
    action = agent.act_greedy(Q_target.predict(state.squeeze(-1)).detach().numpy())
    next_state, reward, reached_goal = env.step(state, agent.actions[action])
    states_for_plotting.append(state.detach().numpy())

    if reached_goal:
        break

    state = next_state

env.plot_states(states_for_plotting)

    