import os
import numpy as np
import matplotlib.pyplot as plt
from envs.environment import Environment
from matplotlib.animation import FuncAnimation


class CartPole(Environment):
    def __init__(self, method):
        super().__init__()
        self.state_dim = 4

        # We define the state as position, velocity, rotation, angular velocity x,x_d,\theta, \theta_d
        self.init_state     = np.zeros((self.state_dim,1))
        self.init_state[2]  = np.pi
        self.terminal_state = np.zeros((self.state_dim,1))
        # Cart-pole hyper-parameters
        self.M               = 1.0
        self.m               = 0.1
        self.l               = 0.5
        self.g               = 9.8
        self.dt              = 0.02 # 50 Hz
        self.reached_goal    = False
        self.reach_threshold = 0.05
        self.x_lim           = 2   # We are limiting the x to be between a thershold to prevent going to infinity

        self.save_dir        = f'./logs/cartPole_{method}'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir+'/animation', exist_ok=True)


    def reset(self):
        self.current_state = self.init_state
        self.reached_goal  = False

    def step(self, state, action):
        state_d = np.zeros((self.state_dim, 1))
        theta = state[2]
        q     = (action + self.m*self.l*np.sin(theta)*state[3]**2)/(self.M + self.m)
        theta_dd = (self.g*np.sin(theta)-q*np.cos(theta))/(self.l*(4/3-((self.m*np.cos(theta)**2)/(self.m+self.M))))
        state_d[0] = state[1]
        state_d[1] = q - (self.m*self.l*np.cos(theta)*theta_dd)/(self.m+self.M)
        state_d[2] = state[3]
        state_d[3] = theta_dd
        next_state = state + self.dt * state_d
        
        # To make sure we remain between [0, 360]
        next_state[2] = next_state[2] % (2*np.pi)
        
        # We want to make the pole upright. It is negative, because we want to maximize
        if next_state[2] < np.pi:
            reward = -np.linalg.norm(next_state-self.terminal_state)
        else:
            diff_    = next_state-self.terminal_state
            diff_[2] = 2*np.pi - next_state[2]
            reward = -np.linalg.norm(diff_)
        if -reward < self.reach_threshold or next_state[0] > self.x_lim or next_state[0] < -self.x_lim:
            self.reached_goal = True
        return next_state, reward, self.reached_goal
    
    def plot_states(self, states):
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

        ani.save(f'{self.save_dir}/animation/cartPole.gif', writer="pillow", fps=2)
        plt.close(fig)
        plt.close()

        
