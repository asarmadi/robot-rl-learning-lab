# Deep Q-Network (DQN)
Instead of keeping track of visited states and corresponding action-values, we use a neural network to predict the action-values. The main reasons is that in large state spaces, it is not feasible to keep track of everything in a table.

The whole idea of Q learning is to learn the action-value function through experineces. So each time, we visit a state, we can update the corresponding action-value in an incremental manner like this:

$$

Q(S,A) = Q(S,A) + \alpha [R + \gamma max_a Q(S',a) - Q(S,A)]

$$

This equation tells us $R + \gamma max_a Q(S',a)$ is the target we want to reach. Therefore, we calculate the error between current estimate $Q(S,A)$ and the target, then with a step size of $\alpha$, we update the action-value function for that state.

DQN uses the same idea. It uses a network called online network $Q_{online}$ to rollout in the environment and collect new experiences. The set of collected experiences is then used to train a network called target $Q_{target}$. This network is the one being used as the network for calculating the target values during training. $Q_{target}$ is to reduce the effect of moving target. If we use the same online network, we keep chasing the target that is being updated by updating the online network. It is like chasing your own tail.

$$

R + \gamma max_{a'} Q_{target}(S',a')

$$

# Double DQN
The max operator for calculating the target values results in an overestimated bias. This means if the target network estimates a high value for a specific action due to noise, DQN selects that action. Double DQN tries to solve this by separating the max operator into two steps: 1) Selecting the index of the maximum 2) Estimating the value. For finding the action, it uses the online network to find the maximum action (it makes sense since the online network is updated on more seen samples). However, for the values it use the target network (therefore, if the max value is high, it won't get amped):

$$

R + \gamma Q_{target}(S', \argmax_{a'}Q_{online}(S',a'))

$$

# Observations
One of the main limitations of DQN style algorithms is that they are working on discrete action space. Therefore, for most of the robotics problems that we deal with continous actions, we need to discretize the actions.

We keep a replay buffer of a constant size and we add the new samples like FIFO. Each time also, we train only on a batch of samples randomly selected from this buffer.