import os
import numpy as np
import matplotlib.pyplot as plt

class BellmanGridWorld2D:
    def __init__(self, grid_size):
        self.out_dir = './logs/GridWorld2D/stateValues/'
        os.makedirs(self.out_dir, exist_ok=True)
        self.grid_size = grid_size

    def plot(self, state_values, iteration):
        fig, ax = plt.subplots()


        ax.imshow(state_values,
                  origin="lower",
                  extent=(0, self.grid_size, 0, self.grid_size)
                  )

        # Display each value inside its cell
        for row in range(state_values.shape[0]):
            for col in range(state_values.shape[1]):
                ax.text(
                    col+0.5,
                    row+0.5,
                    f"{state_values[row, col]:.2f}",
                    ha="center",
                    va="center"
                )

        ax.set_title("State Values")
        ax.set_xticks(np.arange(0, self.grid_size+1, 1))
        ax.set_yticks(np.arange(0, self.grid_size+1, 1))
        ax.grid()
        ax.tick_params()
        fig.savefig(f'{self.out_dir}/{iteration}.png')
        plt.close(fig)
        plt.close()

        