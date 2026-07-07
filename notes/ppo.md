# Proximal Policy Optimization (PPO)
It is an actor critic based method. PPO is improving the A2C by making sure the updated policy won't be too far away from the previous policy. PPO does this by defining a ratio between old (the one used to collect data) and new policy (the one is being trained on the data):

$$
r_{\theta}(t) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

