# Policy Evaluation
In the policy evaluation, the goal is to find the estimate of states values for the given policy.

# Policy Improvement
The goal of policy improvement is to find a better performing policy compared to the given one.

# Policy Iteration
The goal of policy iteration is to find an optimal policy by first evaluating a policy to get a sense of state values. Then using the state values, it updates the policy in a  greedy manner which means the policy will choose an action that results in an immediate maximum of the state values. Once the policy is updated, then again we do the policy evaluation to get a sense of the current policy performance. We do this iteratively to converge to an optimal policy.

The main issue with policy iteration is that it is not computationally effecient. At the value iteration step, we have to update the state values for all the states in the state space till they converge. Then in the improvement step, we sweep again all the states to find the optimal policy.

# Value Iteration
Value iteration combines the two state sweeps into one. Let's look again into the Bellman equation:

$$

v_{\pi}(s) = \sum_{a}\pi(a|s) \sum_{s',r} p(s',r|s,a)[r+\gamma V(s')],~for~all~s\in \mathbb{S}

$$

The optimal state-value function is defined as 

$$

v_{*}(s) = \max_{\pi}v_{\pi}(s)

$$

The Bellman optimality equation expresses the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state following the optimal policy. Therefore,

$$
\begin{aligned}
v_{*}(s) &= \max_{a} \mathbb{E}_{\pi_{*}}[R_{t+1}+\gamma G_{t+1}|S_t=s,A_t=a] \\
         &= \max_{a} \mathbb{E}[R_{t+1}+\gamma v_{*}s(t+1)|S_t=s,A_t=a] \\
         &= \max_{a} \sum_{s',r} p(s',r|s,a)[r+\gamma v_{*}(s')]
\end{aligned}
$$

Based on the definition, the value iteration algorithm treat this equation as an update rule that converges to the optimal state-value function. To be precise, the previous equation uses the optimal state-value function in the inner sum. However, value iteration considers the current state-value function as optimal and updates the state-value function:

$$

v_{k+1}(s) = \max_{a} \sum_{s',r} p(s',r|s,a)[r+\gamma v_k(s')]

$$

for all $s\inS$ and an arbitrary $v_0$.