# Deep Deterministic Policy Gradient (DDPG)
DDPG is similar to DQN with two main differences. First, instead of an epsilon greedy policy, it uses a deep network to estimate the action. Second, the action is continuous. In DQN, the output size of the Q network was the number of action bins. However, in DDPG, the action is an input to the network. DDPG is similar to REINFORCE, A2C, and PPO in a sense that it has a deep network for its policy. The main difference is that it does not generate a probability distribution for the action. DDPG outputs the action directly. 

DDPG is considered as an Actor Critic algorithm. The critic is the action value function. The main difference here is that the action value function outputs a single value compared to DQN that the action value outputs many values for each action bin. That is because we are dealing with a continuous range of action instead of a discrete one. For the same reason we cannot use the same estimation for the current action value like DQN:

$$
y=r+\gamma \max_{a'}Q(s',a')
$$

In a discrete domain, Q outputs m different values and we pick the maximum one. However, in a continuous space, we cannot do that. Instead, we use an actor network to output $a'$ and use that in the action value function to find the DDPG target:

$$
y=r+\gamma Q(s', \mu_{\theta}(s'))
$$

The actor tries to create an action to maximize the action values for a given state:

$$
J^*(\theta)=\max_{\theta}\mathbb{E}_{s\from D}[Q_{\Phi}(s,\mu_{\theta}(s))]
$$

We can solve this maximization by utilizing the MC method over a batch of samples with size B. Therefore, the loss function for minimization would be:

$$
L=-\frac{1}{B}\sum_{i=0}^{B}Q_{\Phi}(s_i,\mu_{\theta}(s_i))
$$

Similar to DQN, since we are bootstraping the target value, the moving target could result in unstable training. To prevent that, DDPG is using target networks for return estimation. Another trick is to slowly update the target network from the online network. In DQN, the online network is copied to the target network after some iterations, In DDPG, this process is done softly:

$$
\theta \leftarrow \tau \theta + (1-\tau)\theta
$$

$$
\Phi \leftarrow \tau \Phi + (1-\tau)\Phi
$$