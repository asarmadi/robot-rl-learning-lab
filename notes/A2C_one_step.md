# One Step Actor Critic (Advantage Actor Critic A2C)
This algorithm bootstraps to get a estimation for the return. Therefore, it won't wait till the end of the episode to update. It uses the current estimate and updates at each step. This method expands on the idea of the REINFORCE with baseline. Instead of waiting till the end of episode to be able to calculate the return and do the update, it bootstraps using the state value function to calculate the TD error. TD error is defined as

$$
\delta_t = r_{t} + \gamma V_{\phi}(s_{t+1}) - V_{\Phi}(s_t)
$$

This is very similar to advantage in REINFORCE that tells us how much our current return is better or worse than the expected return from that state.

Actor-critic has two familiar components. The actor is the policy and the critic evaluates the performance of the actor which is the state value function. 

The update for the actor is similar to the policy update for the REINFORCE:

$$
\theta \leftarrow \theta + \alpha_{\theta} \delta_t \nabla_{\theta} log \pi_{\theta}(a_t|s_t)
$$

The update for the critic is

$$
\Phi \leftarrow \Phi + \alpha_{\Phi} \delta_t \nabla_{\Phi} V_{\Phi}(s_t)
$$

# Observation
I observed that the policy became biased after few episodes. I realized reducing the learnig rate can solve that issue. We should not make the network overfit on those samples. A2C still has the exploration issue.