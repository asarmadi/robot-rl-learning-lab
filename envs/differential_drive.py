import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from envs.environment import Environment
from matplotlib.animation import FuncAnimation


class DifferentialDrive(Environment):
    def __init__(self, method):
        super().__init__()
        self.state_dim  = 3 # x, y, theta

        # We define the state as position, velocity, rotation, angular velocity x,x_d,\theta, \theta_d
        self.init_state     = torch.tensor([1,1,0])

        self.terminal_state = torch.tensor([5,5,0])
        # Cart-pole hyper-parameters
        self.r    = 0.05 # in meters
        self.L    = 0.3 # in meters
        self.dt   = 0.02  # This is 50 Hz
        self.max_steps = 2000 # To terminate the rollout if it takes more than this number of steps
        self.x_lim  = 7 # The environment maximum x limit
        self.y_lim  = 7 # The environment maximum y limit
        self.obstacle_x_min, self.obstacle_x_max = 2, 4 # The boundary values for the obstacle on the x axis
        self.obstacle_y_min, self.obstacle_y_max = 2, 4 # The boundary values for the obstacle on the y axis
        self.terminal_reward = 2000 # This is the reward for terminal cases to be added to the reward

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
        theta       = state[2]
        delta_theta = (delta_s_r-delta_s_l)/self.L

        state_d[0] = delta_s * torch.cos(theta+delta_theta/2)
        state_d[1] = delta_s * torch.sin(theta+delta_theta/2)
        state_d[2] = delta_theta

        next_state = state + state_d

        next_state[2] = torch.atan2(torch.sin(next_state[2]), torch.cos(next_state[2])) # Wrapping the angle to fall whithin -pi to pi

        reward = -torch.linalg.norm(next_state[0:2]-self.terminal_state[0:2])  # The distance to terminal state in x,y
        reward -= 0.001*torch.linalg.norm(action)        # To make sure, the robot excerts minimal force
        reward -= torch.tensor(0.02)  # To punish staying at the same location
        

        terminate = 'running'
        # For the cases that the agent goes out of range on x
        if next_state[0] > self.x_lim or next_state[0] < 0:
            reward -= self.terminal_reward
            terminate = 'terminal'
            print('Out of x bound')


        # For the cases where the agent goes out of range on y
        if next_state[1] > self.y_lim or next_state[1] < 0:
            reward -= self.terminal_reward
            terminate = 'terminal'
            print('Out of y bound')

        # For the final state where the robot reaches the goal/terminal state
        if torch.linalg.norm(next_state[0:2] - self.terminal_state[0:2]) < 0.1:
            reward += self.terminal_reward
            terminate = 'terminal'
            print('Reached goal')

            
        # For the case that the robot hits the obstacle
        if next_state[0] > self.obstacle_x_min and next_state[0] < self.obstacle_x_max and \
            next_state[1] > self.obstacle_y_min and next_state[1] < self.obstacle_y_max:
            reward -= self.terminal_reward
            terminate = 'terminal'
            print('Hit obstacle')


        # The time-limit
        if terminate == 'running' and step_counter >= self.max_steps:
            terminate = 'truncate'
            print('Out of time')

        
        return next_state, reward, terminate
    
    def plot_states(self, states, actions, name_str=''):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.x_lim)
        ax.set_ylim(0, self.y_lim)

        x, y, theta = self.init_state[0].cpu().numpy(), self.init_state[1].cpu().numpy(), self.init_state[2].cpu().numpy()

        axis_,   = ax.plot([x+self.L*np.sin(theta), x-self.L*np.sin(theta)], [y-self.L*np.cos(theta), y+self.L*np.cos(theta)], "b--", markersize=20)
        wheel_r, = ax.plot([x+self.L*np.sin(theta)+self.r*np.cos(theta), x+self.L*np.sin(theta)-self.r*np.cos(theta)], [y-self.L*np.cos(theta)+self.r*np.sin(theta), y-self.L*np.cos(theta)-+self.r*np.sin(theta)], "g-", markersize=20)
        wheel_l, = ax.plot([x-self.L*np.sin(theta)+self.r*np.cos(theta), x-self.L*np.sin(theta)-self.r*np.cos(theta)], [y+self.L*np.cos(theta)+self.r*np.sin(theta), y+self.L*np.cos(theta)-+self.r*np.sin(theta)], "b-", markersize=20)

        obstacle, = ax.plot([self.obstacle_x_min,self.obstacle_x_max,self.obstacle_x_max,self.obstacle_x_min,self.obstacle_x_min], \
                            [self.obstacle_y_max,self.obstacle_y_max,self.obstacle_y_min,self.obstacle_y_min,self.obstacle_y_max], "r-", markersize=20)

        terminal_, = ax.plot([self.terminal_state[0].cpu().numpy()],[self.terminal_state[1].cpu().numpy()], 'c*', markersize=20)
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
                squashed_action = torch.tanh(output[0:2])       
                action_env = policy.max_action * squashed_action 
            next_state, reward, terminate = self.step(state, action_env, step)
            states_for_plotting.append(state.detach().cpu().numpy())
            actions_for_plotting.append(action_env.cpu().numpy())

            if terminate == 'terminal' or terminate == 'truncate':
                break
            
            state = next_state
            step += 1
        self.plot_states(states_for_plotting, actions_for_plotting, name_str=episode)


        
