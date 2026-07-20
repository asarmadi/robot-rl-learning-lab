import torch
from envs.cart_pole import CartPole
from agents.epsilon_greedy import epsilonGreedy
from estimators.mlp import MLP
from utils.training import Training
from utils.replay_buffer import ReplayBuffer

seed              = 0
torch.manual_seed(seed)

# DQN Hyper-Parameters
action_type        = 'epsilonGreedy'
n_episodes         = 10000
epsilon            = 1.0  # Probability for taking random actions
action_dim         = 10   # Number of bins for the action space
gamma              = 0.9  # Discount factor
initial_data_size  = 1000 # This is used to make sure, we train the network after dataset has some samples and not train only on few samples in the begining
update_rate        = 4 # Number of steps used to copy the online network to the target network.
replay_buffer_size = 50000
max_action         = 5

# Training Hyper-Parameters
batch_size        = 128
lr                = 0.0001

# Q Neural Networks Hyper-Parameters
hidden_dim        = 128
n_layers          = 2
device            = 'cpu'





env      = CartPole(method='DQN')
agent    = epsilonGreedy(epsilon_max=epsilon, action_dim=action_dim, seed=seed, max_action=max_action, type_=action_type, environment='CartPole')
Q_online = MLP(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=agent.action_dim)
Q_target = MLP(input_dim=env.state_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=agent.action_dim)

training = Training(batch_size=batch_size,
                    lr=lr,
                    gamma=gamma,
                    device=device,
                    target_network=Q_target,
                    online_network=Q_online)

replay_buffer = ReplayBuffer(size=replay_buffer_size,state_dim=env.state_dim)

total_steps = 1
rewards = []

for episode in range(n_episodes):
    env.reset()
    state = env.current_state
    sum_rewards = 0
    step  = 0

    # Added this to start with high exploration and then move towards exploitation
    agent.set_epsilon(total_steps)

    while True:
        action = agent.act(Q_online.predict(state).detach().numpy())
        next_state, reward, terminate = env.step(state, agent.actions[action], step_counter=step)
        replay_buffer.additem(state,next_state,reward,action)

        if terminate == 'terminal' or terminate == 'truncate':
            break

        step += 1
        total_steps += 1
        sum_rewards += reward
        state = next_state
        
        if (len(replay_buffer) > initial_data_size): # We update the online network after collecting inital number of samples
            training.train(replay_buffer, copy_network=(step % update_rate))

    rewards.append(sum_rewards)
    print(f'Episode: {episode}, Steps: {step}, Reward: {sum_rewards}')

    # plotting the result every 1000 episodes
    if episode % 200 == 0:
        env.test_policy(agent, episode//200, Q_network=Q_online)
        env.plot_rewards(rewards)

    