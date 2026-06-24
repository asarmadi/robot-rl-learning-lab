# Deep Q-Network (DQN)
Instead of keeping track of visited states and corresponding action-values, we use a neural network to predict the action-values. The main reasons is that in large state spaces, it is not feasible to keep track of everything in a table.

The whole idea of Q learning is to learn the action-value function through experineces. So each time, we visit a state, we can update the corresponding action-value in an incremental manner like this:

$$

Q(S,A) = Q(S,A) + \alpha [R + \gamma max_a Q(S',a) - Q(S,A)]

$$

This equation tells us $R + \gamma max_a Q(S',a)$ is the target we want to reach. Therefore, we calculate the error between current estimate $Q(S,A)$ and the target, then with a step size of $\alpha$, we update the action-value function for that state.

DQN uses the same idea. It uses a network called online network $Q_{online}$ to rollout in the environment and collect new experiences. The set of collected experiences is then used to train a network called target $Q_{target}$. This network is the one being used as the network for calculating the target values during training.

$$

R + \gamma max_a Q_{target}(S',a)

$$

# Implementation
We can use only one network called $Q_{online}$ to collect samples and occasionally copy the weights of this network to another network called target that is being used to create the target values during training.