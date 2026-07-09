import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from envs.environment import Environment
from matplotlib.animation import FuncAnimation


class CartPole(Environment):
    def __init__(self, method):
        super().__init__()
        self.state_dim  = 4

        # We define the state as position, velocity, rotation, angular velocity x,x_d,\theta, \theta_d
        self.init_state     = torch.zeros((self.state_dim,1))
        self.init_state[2]  = torch.pi/18
        self.terminal_state = torch.zeros((self.state_dim,1))
        # Cart-pole hyper-parameters
        self.M               = 1.0
        self.m               = 0.1
        self.l               = 0.5
        self.g               = 9.8
        self.dt              = 0.02 # 50 Hz
        self.terminate       = 'running'
        self.reach_threshold = 2*(torch.pi/180) # 2 Degree boundary
        self.x_lim           = 2   # We are limiting the x to be between a thershold to prevent going to infinity
        self.upright_counter = 0   # TO keep track of the agent when it is stable upright
        self.upright_threshold = 200 # Number of steps that we consider the pole to be upright
        self.max_steps         = 6000 # TO prevent the running loop stays forever


        self.save_dir        = f'./logs/cartPole_{method}'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir+'/animation', exist_ok=True)


    def reset(self):
        self.current_state   = self.init_state
        self.terminate       = 'running'
        self.upright_counter = 0

    def step(self, state, action, step_counter):
        state_d = torch.zeros((self.state_dim, 1))
        theta = state[2]
        q     = (action + self.m*self.l*torch.sin(theta)*state[3]**2)/(self.M + self.m)
        theta_dd = (self.g*torch.sin(theta)-q*torch.cos(theta))/(self.l*(4/3-((self.m*torch.cos(theta)**2)/(self.m+self.M))))
        state_d[0] = state[1]
        state_d[1] = q - (self.m*self.l*torch.cos(theta)*theta_dd)/(self.m+self.M)
        state_d[2] = state[3]
        state_d[3] = theta_dd
        next_state = state + self.dt * state_d

        reward = 0
        
        # We want to make the pole upright. It is negative, because we want to maximize
        if abs(next_state[2,0]) <= self.reach_threshold:
            reward = 1
            self.upright_counter += 1
        else:
            self.upright_counter = 0 # To make sure if the pole is out of the upright poisiton, we reset
            reward = -1

        # For the cases that the agent goes to infinity on x
        if next_state[0,0] > self.x_lim or next_state[0,0] < -self.x_lim:
            self.terminate = 'terminal'
        # To encourage the agent to learn to stay at the upright position
        if self.upright_counter >= self.upright_threshold:
            self.terminate = 'terminal'

        if step_counter >= self.max_steps:
            self.terminate = 'truncate'
        
        return next_state, reward, self.terminate
    
    def plot_states(self, states, actions, name_str=''):
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        pole,  = ax.plot([0, 0], [0, -self.l], "--", markersize=20)
        cart,  = ax.plot([0.05, 0.05, -0.05, -0.05, 0.05], [-0.05, 0.05, 0.05, -0.05, -0.05], "-", markersize=20)



        def update_frame(frame):
            x = states[frame][0].item()
            theta = states[frame][2].item()
            pole.set_data([x, x+self.l*np.sin(theta)], [0, self.l*np.cos(theta)])
            cart.set_data([x+0.05, x+0.05, x-0.05, x-0.05, x+0.05],[-0.05, 0.05, 0.05, -0.05, -0.05])
            return pole,cart

        ani = FuncAnimation(
            fig,
            update_frame,
            frames=len(states),
            interval=self.dt*1000,
            blit=True
        )

        ani.save(f'{self.save_dir}/animation/cartPole{name_str}.gif', writer="pillow", fps=int(1 / self.dt))
        plt.close(fig)
        plt.close()

        fig, axes = plt.subplots(5, 1)

        state_names = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']

        for i in range(self.state_dim):
            axes[i].plot(np.array(states)[:,i], label=state_names[i])
            axes[i].legend()

        axes[self.state_dim].plot(np.array(actions), label='F')
        axes[self.state_dim].legend()

        plt.savefig(f'{self.save_dir}/states.png')
        plt.close(fig)
        plt.close()

    def test_policy(self, policy, episode):
        self.reset()
        state = self.current_state
        states_for_plotting = []
        actions_for_plotting = []

        step = 0

        while True:
            logits = policy.predict(state.squeeze(-1)).detach()
            action = logits.argmax()
            next_state, reward, terminate = self.step(state, policy.actions[action], step)
            states_for_plotting.append(state.detach().numpy())
            actions_for_plotting.append(policy.actions[action])

            if terminate == 'terminal' or terminate == 'truncate':
                break
            
            state = next_state
            step += 1

        self.plot_states(states_for_plotting, actions_for_plotting, name_str=episode)


        
