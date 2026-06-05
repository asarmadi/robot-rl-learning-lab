# The One-Dimensional Random Walk
The agent walks along a line. The agent could move one step to the right or left at each time step.Each pace hase the same length.

## Problem

We have a 1D line with 7 states (i.e, -3, -2, -1, 0, 1, 2, 3). In the beginning of each episode, the agent starts from the middle (i.e, 0). The episode ends if the agent moves to far right (i.e, 3) or far left (i.e., -3). The agent receives a reward of 1 for the last step on right (i.e., 3). It receives 0 reward for any other state.

## Policy

The agent uses a random policy at each step. Before each step, the agent flips a coin. If it is heads, it takes one step forward. If it is tails, it takes one step back.

## Return
We are using Monte Carlo return estimator to update the return value after each episode. Return at time t is defined as 
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R{t+3} + \cdots
$$