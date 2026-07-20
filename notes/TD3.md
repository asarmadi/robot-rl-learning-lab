# Twin Delayed Deep Deterministic Policy Gradient (TD3)
This algorithm is an improved version of DDPG and as the name suggests it improved DDPG in two ways. First, instead of having only one critic, it uses two separate critics to estimate the target value for the critic. To have pessimistic estimation, it considers the minimum of the two critics:

$$
Q=\min(Q_{\theta_1}(s',a'),Q_{\theta_2}(s',a'))
$$

Then this is used to find the traget value to train both network:

$$
y=r+\gamma (1-d) Q
$$

where $d$ is 1 when the current state (i.e., $s$) is the terminal state. TD3 also adds noise during critic target construction to prevent the critic to assign hight value to a particular action. Instead, it tries to make sure the actions near that specific point is still valuable or not.

The second change compared to DDPG is to update the actor with a delay compared to the critic.