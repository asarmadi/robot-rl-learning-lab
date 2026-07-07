# Incremental Implementation of MC
Monte Carlo (MC) is estimating the state-value values by rolling out the policy and taking the average of the returns for the state being seen in that episode. MC is runing this iteration for multiple times to get a good estimate of the sate-values. MC is using the following equation for estimating state-value table

$$
v(s) = \frac{1}{N+M} \sum_{k=0}^{N} \sum_{m=0}^{M_s} G_k^{m}(s)
$$

where $N$ is the total number of iterations, $M_s$ is the number of observations of state $s$ in iteration $k$ and $G_k^{m}$ is the reward value for state $s$ in $k^{th}$ episode for the $m^{th}$.

The average could be updated in an incremental manner which is the heart of the TD estimation:

$$
V(s_n) = V(s_n) + \alpha [G_n - V(s_n)]
$$

where $G_n$ is the observed return at iteration $n$. As you can see, the state-value can be updated once the return $G_n$ for that state is known. This happens in the end of the iteration, since we then can calculate the $G_n$ moving backwards.

# Temporal Difference (TD) Estimation
Unlike MC that we need to wait till the end of the episode, TD is using an estimate of the state-value function to update the state-value once the agent visits a state:

$$
V(s_n) = V(s_n) + \alpha [R+\gamma V(s_{n+1}) - V(s_n)]
$$

As you can see, here instead of the actual return value, we use an estimate of the return $(R+\gamma V(s_{n+1}))$.

# Sarsa: On-policy TD Control
The goal of Sarsa is to estimate the action-value using the TD error:

$$
Q(S,A) = Q(S,A) + \alpha [R + \gamma Q(S',A') - Q(S,A)]
$$

The reason we call Sarsa as on-policy because it is using the action-value in an $\epsilon$-greedy manner and then uses the same action-value to update the action-value.


# Q-learning: Off-policy TD Control
The goal of Q-learning is learning the optimal action-values:

$$
Q(S,A) = Q(S,A) + \alpha [R + \gamma max_a Q(S',a) - Q(S,A)]
$$

Q-learning is considered as off-policy since it is using $\epsilon$-greedy policy to take action in each state, however, it uses the optimal action-value for updating the action-value.