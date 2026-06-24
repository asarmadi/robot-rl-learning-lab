from envs.cart_pole import CartPole
from agents.epsilon_greedy import epsilonGreedy
from estimators.mlp import MLP

seed              = 0
n_posides         = 100
n_hidden          = 2
n_nodes           = 16
epsilon           = 0.1
initial_data_size = 1000 # This is used to make sure, we train the network after dataset has some samples and not train only on few samples in the begining
update_rate       = 1000 # Number of steps used to copy the online network to the target network.

env      = CartPole()
agent    = epsilonGreedy(epsilon=epsilon)
Q_online = MLP(n_hidden=n_hidden, n_nodes=n_nodes, output_dim=env.action_dim)


for episode in range(n_episodes):

    env.reset()
    state = env.current_state
    step = 0

    while True:
        action = agent.act(state, Q_online)
        next_state, reward = env.step(state, action)
        replay_buffer.append(state, action, new_state, reward)

        if (next_state == env.terminal_state).all():
            break

        step += 1

        if (replay_buffer.size > initial_data_size): # We update the online network after collecting inital number of samples
            Q_online.train_network(replay_buffer, Q_online, copy_network=(step % update_rate))


    