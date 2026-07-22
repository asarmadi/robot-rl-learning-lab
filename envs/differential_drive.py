import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from envs.environment import Environment
from matplotlib.animation import FuncAnimation


class CartPole(Environment):
    def __init__(self, method):
        super().__init__()
        self.state_dim  = 3 # x, y, theta

        # We define the state as position, velocity, rotation, angular velocity x,x_d,\theta, \theta_d
        self.init_state     = torch.zeros(self.state_dim)

        self.terminal_state = torch.tensor([2,2,0])
        # Cart-pole hyper-parameters
        self.r    = 
        self.L    =
        self.max_steps = 1000 # To terminate the rollout if it takes more than this number of steps
        self.x_lim  = 3 # The environment maximum x limit
        self.y_lim  = 3 # The environment maximum y limit

        self.save_dir        = f'./logs/differentialDrive_{method}'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir+'/animation', exist_ok=True)


    def reset(self):
        self.current_state   = self.init_state

    def step(self, state, action, step_counter):
        state_d = torch.zeros(self.state_dim)
        # action[0] is the right wheel angular speed
        # action[1] is the left wheel angular speed
        delta_s_r   = self.r * action[0] * self.dt
        delta_s_l   = self.r * action[1] * self.dt
        delta_s     = (delta_s_r+delta_s_l)/2
        theta       = state[0]
        delta_theta = (delta_s_r-delta_s_l)/self.L

        state_d[0] = delta_s * torch.cos(theta+delta_theta/2)
        state_d[1] = delta_s * torch.sin(theta+delta_theta/2)
        state_d[2] = delta_theta

        next_state = state + state_d

        reward = torch.linalg.norm(next_state-self.terminal_state)  # The distance to terminal state

        terminate = 'running'
        # For the cases that the agent goes out of range on x
        if next_state[0] > self.x_lim or next_state[0] < 0:
            terminate = 'terminal'

        # For the cases where the agent goes out of range on y
        if next_state[1] > self.y_lim or next_state[1] < 0:
            terminate = 'terminal'

        if step_counter >= self.max_steps:
            terminate = 'truncate'
        
        return next_state, reward, terminate
    
    def plot_states(self, states, actions, name_str=''):
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        x, y, theta = self.init_state[0], self.init_state[1], self.init_state[2]

        axis_,   = ax.plot([x+self.L*np.sin(theta), x-self.L*np.sin(theta)], [y-self.L*np.cos(theta), y+self.L*np.cos(theta)], "--", markersize=20)
        wheel_r, = ax.plot([x+self.L*np.sin(theta)+self.r*np.cos(theta), x+self.L*np.sin(theta)-self.r*np.cos(theta)], [y-self.L*np.cos(theta)+self.r*np.sin(theta), y-self.L*np.cos(theta)-+self.r*np.sin(theta)], "-", markersize=20)
        wheel_l, = ax.plot([x-self.L*np.sin(theta)+self.r*np.cos(theta), x-self.L*np.sin(theta)-self.r*np.cos(theta)], [y+self.L*np.cos(theta)+self.r*np.sin(theta), y+self.L*np.cos(theta)-+self.r*np.sin(theta)], "-", markersize=20)

        def update_frame(frame):
            x     = states[frame][0].item()
            y     = states[frame][1].item()
            theta = states[frame][2].item()
            axis_.set_data([x+self.L*np.sin(theta), x-self.L*np.sin(theta)], [y-self.L*np.cos(theta), y+self.L*np.cos(theta)])
            wheel_r.set_data([x+self.L*np.sin(theta)+self.r*np.cos(theta), x+self.L*np.sin(theta)-self.r*np.cos(theta)], [y-self.L*np.cos(theta)+self.r*np.sin(theta), y-self.L*np.cos(theta)-+self.r*np.sin(theta)])
            wheel_l.set_data([x-self.L*np.sin(theta)+self.r*np.cos(theta), x-self.L*np.sin(theta)-self.r*np.cos(theta)], [y+self.L*np.cos(theta)+self.r*np.sin(theta), y+self.L*np.cos(theta)-+self.r*np.sin(theta)])
            return axis_, wheel_r, wheel_l

        ani = FuncAnimation(
            fig,
            update_frame,
            frames=len(states),
            interval=self.dt*1000, # Multiplied to 1000 to convert s to ms
            blit=True
        )

        ani.save(f'{self.save_dir}/animation/differentialDrive{name_str}.gif', writer="pillow", fps=int(1 / self.dt))
        plt.close(fig)
        plt.close()

        fig, axes = plt.subplots(5, 1)

        state_names = [r'$x$', r'$y$', r'$\theta$']

        for i in range(self.state_dim):
            axes[i].plot(np.array(states)[:,i], label=state_names[i])
            axes[i].legend()

        action_names = [r'$\omega_R$', r'$\omega_L$']

        for i in range(2):
            axes[self.state_dim+i].plot(np.array(actions)[:,i], label=action_names[i])
            axes[self.state_dim+i].legend()

        plt.savefig(f'{self.save_dir}/states.png')
        plt.close(fig)
        plt.close()

    def plot_rewards(self, rewards):
        plt.figure()

        state_names = [r'Reward']

        plt.plot(rewards)

        plt.title('Cumulative Reward')

        plt.savefig(f'{self.save_dir}/rewards.png')
        plt.close()


    def test_policy(self, policy, episode, Q_network=None):
        self.reset()
        state = self.current_state
        states_for_plotting = []
        actions_for_plotting = []

        step = 0

        while True:
            if policy.type == 'discrete':
                logits = policy.predict(state).detach()
                action = logits.argmax()
                action_env = policy.actions[action]
            elif policy.type == 'epsilonGreedy':
                action = torch.argmax(Q_network(state).detach())
                action_env = torch.tensor(policy.actions[action.item()])
            else:
                output = policy.predict(state).detach()
                #squashed_action = torch.tanh(output[0])       
                action_env = policy.max_action * output[0] 
            next_state, reward, terminate = self.step(state, action_env, step)
            states_for_plotting.append(state.detach().numpy())
            actions_for_plotting.append(action_env)

            if terminate == 'terminal' or terminate == 'truncate':
                break
            
            state = next_state
            step += 1

        self.plot_states(states_for_plotting, actions_for_plotting, name_str=episode)


        
