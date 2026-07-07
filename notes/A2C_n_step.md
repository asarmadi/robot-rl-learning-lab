# Batch n-step A2C
A2C is using the actor-critic framework in which the actor is trained on the one step advantage and the critic is trained using the one step bootstraped return value. N-step A2C combines the idea of bootstraping with full return. Instead of finding the full return, it calculates the return for n future states and bootsrtaps from that state.

In the one step A2C the estimation for the return value is

$$
r_t + \gamma V(s_{t+1})
$$

In n-step A2C, instead of considering only one step ahead, we look more than one steps ahead and use the actual rewards to calculate the return and from the $n^{th}$ step on we use the state value estimation:

$$
r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{n-1}r_{t+n-1}+\gamma^n V(s_{t+n})
$$

We can rewrite this as

$$
G_t^n = \sum_{k=0}^{n-1} \gamma^k r_{t+k}+\gamma^n V(s_{t+n})
$$

The rest of the algorithm is same as A2C. There is another term that is added to the actor loss to encourage the exploration. We had the exploration issue from REINFORCE that, the policy can easily become biased towards the early samples it sees. The only mechanism then was the stochasticity of the policy. Earlier algorithms like DQN, they use $\epsilon$-greedy policy to explore. In n-step A2C, an entropy regulizer is added to the actor loss to make sure we increase the entropy, therefore, the policy has the chance to explore more.

# Observation
For the n-step A2C, there are two ways to implement the algorithm. We can keep a buffer of size n and just update for the first state that we already have the future n states. The reason is that the only state that we have the full 5 future states is the first state in the list, but for the last one in the list we still need to calculate the future n states to do the same thing. The other way to implement the algorithm (which seems like is the standard way) is to collect n states and update all of them. In this way, the first state is using n future states, the next one n-1, etc.

The benefit of n-step compared to the one step algorithm is that it has less bias since it is partially using the future rewards to estimate the return instead of bootstraping. However, it increases variance.