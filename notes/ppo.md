# Proximal Policy Optimization (PPO)
It is an actor critic based method. PPO is improving the A2C by making sure the updated policy won't be too far away from the policy used to collect the data. PPO does this by defining a ratio between old (the one used to collect data) and new policy (the one is being trained on the data):

$$
r_{\theta}(t) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

This quantity has an actual meaning which comes from importance sampling. The idea is that we want to calculate the expectation under a probability distribution while the samples are drwan from another distribution. In this case, the expectation is $\mathbb{E}_{a\sim \pi_{new}}A(s,a)$ that we want to maximize. However, our rollout buffer has samples collected using $\pi_{old}$. The question is whether we can reweight the samples coming from the old policy to behave like they are coming from the new policy. Importance sampling proposes rewighting the probability distribution like $\mathbb{E}_{s,a\sim\pi_{old}}\frac{\pi_{new}(s,a)}{\pi_{old}(s,a)}A(s,a)$. Therefore, $r_{\theta}$ is the importance sampling ration.
PPO defines the objective similar to A2C, the differnce is that it uses the newly defined variable $r_{\theta}(t)$ in the log. The reason that it is correct is that, $\pi_{\theta_{old}}$ is not dependent on the $\theta$, therefore, it is like adding a zero to the gradient:

$$
J(\theta) = \mathbb{E}r_{\theta}(t)A_t
$$

The other technique that is used in PPO that makes it different is the clipping which makes sure that the change won't be too much. It prevents the policy to be very different from the current policy. Therefore, the overall policy optimization looks like:

$$
\mathbb{E}\min{(r_{\theta}(t)A_t, clip(r_{\theta}(t), 1-\epsilon, 1+\epsilon)A_t)}
$$

One thing to notice is the min function. The min function makes it even more pessemistic by choosing the minimum value. Our goal is to maximize, therefore, we want to prevent cases that the clipping is actually results in a larger number than the $r_{\theta}(t)A_t$.

The other technique is the way they calculate the return values. It uses Generalized Advantage Estimation (GAE). In the n-step advantage estimation, tried to reduce the bias from one step advantage estimation by using n future rewards to estimate the return. However, it was dependent on the choice of the n. If we set n large, it was closer to the Monte Carlo estimation, and lower n was closer to the TD estimation. GAE uses a weighted n-step advantage estimation to get a better estimation of the return. The idea starts from the one-step TD error:

$$
A_t^1=\delta_t=r_t+\gamma V(s_{t+1}) - V(s_t)
$$

Then

$$
A_t^2=r_t+\gamma r_{t+1} + \gamma^2V(s_{t+2}) - V(s_t)=\delta_t+\gamma \delta_{t+!}
$$

In the same way, the genral formulation is

$$
A_t^n=\sum_{l=0}^{n-1}\gamma^l \delta_{t+l}
$$

GAE calculates the weighted average of many n-step advantages

$$
A_t=(1-\lambda)[A_t^1+\lambda A_t^2+\lambda^2 A_t^3+\cdots]
$$

By substituting the advantage values, we will get the following general formula

$$
A_t=\sum_{l=0}^{T} (\gamma \lambda)^l \delta_{t+l}
$$

where $T$ is the rollout length. The GAE could be calculated in recursive manner as well:

$$
A_t = \delta_t + \gamma \lambda A_{t+1}
$$

The thrid important element is the entropy added to the action loss to encourage exploration.

# Observations
One of the issues that I was facing was getting NaN during training. I realized that adding advantage normalization for the rollout buffer solves the issue.

Another problem was for continuous actions. After a while the policy starts to increase the std. We should clip the log_std to make sure the std does not blow up after a while.

One of the mistakes that I made was using the normalized advantage for training the state value (critic) function. The critic is going to give us the state value and we only normalize for training actor because the direction is important for us.