# REINFORCE with Baseline
This algorithm is similar to the vanila REINFORCE, however, instead of a return it uses advantage which is return minus the value function. The value function gives us a sense of expected return for that state, therefore, we can move in that direction. Since the value function is not dependant on the $\theta$ (the policy weights), it won't change the gradient of the loss, therefore, the new loss will look like this

$$

J(\theta) = - \sum_{t=0}^{T} \pi_{\theta}(a_t|s_t) (G_t - V_{\phi}(s_t))

$$

where $V_{\phi}$ is a separate neural network with $\phi$ as weights. We train the value function on a set of samples collected during an episode to train the model.
