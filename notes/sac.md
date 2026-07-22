# Soft Actor-Critic (SAC)
The idea of the SAC is very similar to the TD3. There are two main differences. First, in SAC the actor generates a probability distribution of the actions while in TD3, the actor directly outputs the action. Second, in SAC the actor goal is not only maximizing the action value, also maximizing the policy entropy. By maximizing the entropy, the policy will explore the state space.

In algorithms like DDPG and TD3, the exploration was performed by adding noise to the actions during actor rollout while collecting samples in a replay buffer. 

In SAC, the objective is

$$
J(\pi) = \mathbh{E}_{\pi} [\sum_{t=0}^{\infty} \gamma^t(r_t+\alpha \mathbh{H}(\pi(.|s_t)))]
$$

where $\mathbh{H}$ is the entropy of the policy and $\alpha$ controls how much we care about exploration. For a probabilistic policy, the entropy could be defined as $\mathbh{H(\pi(.|s))}=-\mathbh{E}_{a\from \pi}[\log{\pi(a|s)}]$, therefore, the objective will be:

$$
J(\pi)=\mathbh{E}_{\pi}[\sum_{t=0}^{\infty}\gamma^t(r_t-\alpha \log{\pi(a_t|s_t)})]
$$

SAC bootstraps the action values, and building on the entropy maximization idea, the action value function estimation becomes:

$$
Q^{\pi}(s_t,a_t)=r_t+\gamma \mathbh{E}[Q^{\pi}(s_{t+1},a_{t+1})-\alpha \log{\pi(a_{t+1}|s_{t+!})}]
$$

The rest of the algorithm is very similar to the TD3 where it is using two critics to reduce the overstimation bias. Also, it uses the soft update idea to slowly update the target networks from the online ones. However, SAC does not consider two networks for the actor. It has only one probabilistic actor that is used to during the target value calculations for the critics.

The other difference wrt TD3 is the actor update objective. In TD3, we only use the online actor to estimate the action for maximazing the action value function. However, for the SAC, we consider the minimum between the two online networks and we also add the entropy maximization term.

# Observations
I realized this issue in SAC, TD3, and DDPG that the action value network (i.e., Q) takes a concatenation of states and action as the input. These may have a very different range. A method similar to PPO advantage noramlization may help.