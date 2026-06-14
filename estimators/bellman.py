import os
import numpy as np
import matplotlib.pyplot as plt

class BellmanGridWorld2D:
    def __init__(self):
        self.out_dir = './logs/GridWorld2D/stateValues/'
        os.makedirs(self.out_dir, exist_ok=True)

    def plot(self, state_values, iteration):
        plt.figure()
        fig, ax = plt.subplots()

        ax.imshow(state_values)

        # Display each value inside its cell
        for row in range(state_values.shape[0]):
            for col in range(state_values.shape[1]):
                ax.text(
                    col,
                    row,
                    f"{state_values[row, col]:.2f}",
                    ha="center",
                    va="center"
                )

        ax.set_xticks(np.arange(state_values.shape[1]))
        ax.set_yticks(np.arange(state_values.shape[0]))

        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title("State values")

        plt.savefig(f'{self.out_dir}{iteration}.png')


        