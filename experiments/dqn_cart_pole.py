from envs.cart_pole import CartPole
from agents.epsilon_greedy import epsilonGreedy
from estimators.mlp import MLP
from utils.training import Training

seed              = 0
n_episode         = 100
hidden_dim        = 2
n_layers          = 16
epsilon           = 0.1
initial_data_size = 1000 # This is used to make sure, we train the network after dataset has some samples and not train only on few samples in the begining
update_rate       = 1000 # Number of steps used to copy the online network to the target network.
n_epochs          = 100
batch_size        = 8


env      = CartPole()
agent    = epsilonGreedy(epsilon=epsilon)
Q_online = MLP(n_hidden=n_hidden, n_layers=n_layers, output_dim=env.action_dim)
Q_target = MLP(n_hidden=n_hidden, n_layers=n_layers, output_dim=env.action_dim)

training = Training(batch_size=batch_size,
                    n_epochs=n_epochs,
                    target_network=Q_target,
                    online_network=Q_online)

for episode in range(n_episodes):

    env.reset()
    state = env.current_state
    step = 0

    while True:
        action = agent.act(state, Q_online)
        next_state, reward = env.step(state, action)
        replay_buffer.additem(state, new_state, reward, action)

        if (next_state == env.terminal_state).all():
            break

        step += 1

        if (replay_buffer.size > initial_data_size): # We update the online network after collecting inital number of samples
            training.train(replay_buffer, copy_network=(step % update_rate))


    