import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MonteCarloValueEstimator:
    def __init__(self, n_states, dir_, gamma=0.9):
        self.n_states       = n_states
        self.returns        = {}
        self.state_values   = {}
        self.episode_states = [] # For making animation purposes
        self.gamma          = gamma
        self.save_dir       = './logs/'+dir_
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir+'state_values/', exist_ok=True)
        os.makedirs(self.save_dir+'state_animation/', exist_ok=True)


    def update(self, episode):
        # episode is a list of (state, reward) pairs that is filled from time step 0 to T
        g = 0 # Initialize the return value
        self.episode_states = []
        for state, reward in episode[::-1]:
            g = self.gamma * g + reward
            self.returns.setdefault(state, []).append(g)
            self.state_values[state] = sum(self.returns[state])/len(self.returns[state])
            self.episode_states.append(state)
        

    def plot_states(self, dir_):

        fig, ax = plt.subplots()
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_xticks([-3,-2,-1,0,1,2,3])

        agent, = ax.plot([], [], "o", markersize=20)

        def update(frame):
            state = self.episode_states[::-1][frame]
            agent.set_data([state], [0])
            return agent,

        ani = FuncAnimation(
            fig,
            update,
            frames=len(self.episode_states),
            interval=500,
            blit=True
        )

        ani.save(f'{self.save_dir}state_animation/{dir_}.gif', writer="pillow", fps=2)


    def plot_value(self, dir_):
        plt.figure()
        plt.bar(self.state_values.keys(), self.state_values.values())
        plt.xlabel("States")
        plt.ylabel("Value")
        plt.title("State-value")
        plt.savefig(f'{self.save_dir}state_values/{dir_}.png')
        plt.close()


