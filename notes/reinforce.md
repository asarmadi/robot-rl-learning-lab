# REINFORCE
The main idea in REINFORCE is to use a neural network as the policy and learn policy directly. It does not learn a value function to use it for choosing the actions. Also, REINFORCE is a using the return value instead of using an estimate for finding the target value. The whole idea of the REINFORCE starts from the fact we want to maximize the expected return:

$$

J(\theta) = \mathbb{E}_{\tau \mathcal p_{\theta}(\tau)} G(\tau)

$$

where:
\begin{itemize}
\item $\theta$ = policy network parameters
\item $\tau$ = full trajectory
\item $\p_{\theta}(\tau)$ = probability of generating the trajectory under the policy
\item $G(\tau)$ = total return of the trajectory
\end{itemize}

$$

J(\theta) = \int p_{\theta}(\tau) G(\tau) \, \mathrm{d}\tau

$$

the gradient is 

$$

\begin{aligned}
\nabla_{\theta}J(\theta) &= \nabla_{\theta} \int p_{\theta}(\tau) G(\tau)  \, \mathrm{d}\tau \\
 &= \int \nabla_{\theta}p_{\theta}(\tau) G(\tau)  \, \mathrm{d}\tau
\end{aligned}

$$

The key is to replace $\nabla_{\theta}p_{\theta}(\tau)=p_{\theta}(\tau)\nabla_{\theta}p_{\theta}(\tau)$, therefore, we have

$$

\begin{aligned}
\nabla_{\theta}J(\theta) &= \int p_{\theta}(\tau)\nabla_{\theta} log p_{\theta}(\tau) G(\tau)  \, \mathrm{d}\tau \\
&= \mathbb{E}_{\tau \mathcal p_{\theta}(\tau)} [\nabla_{\theta} log p_{\theta}(\tau) G(\tau)]
\end{aligned}

$$

We expand the trajectory probability as $p_{\theta}(\tau)=p(s_0) \prod_{t=0}^{t=T}\pi(a_t|s_t) p(s_{t+1}|s_t,a_t)$, therefore, $log p_{\theta}(\tau)=log p(s_0) + \sum_{t=0}^{t=T}log \pi_{\theta}(a_t|s_t)+\sum_{t=0}^{t=T}log p(s_{t+1}|s_t,a_t)$. Now, we need to take the derivative w.r.t $\theta$. You can see only policy is dependent on $\theta$, therefore, $\nabla_{\theta}log p_{\theta}(\tau)=\sum_{t=0}^{t=T} \nabla_{\theta}log \pi_{\theta}(a_t|s_t)$. By subsituting this in the gradient of the expected return, we have

$$

\nabla_{\theta}J(\theta) = \mathbb{E}_{\tau \mathcal p_{\theta}(\tau)}[\sum_{t=0}^{t=T}\nabla_{\theta}log \pi_{\theta}(a_t|s_t)G(\tau)]

$$

There is one point here, the expected return is using the whole trajectory return for the trajectory. We can modify this to return to-go from that specific state. The reason is that the current actions does not depend on the previous rewards. We can also do this replacement since the rewards before time $t$ does not depend on the $a_t$. TO ME IT IS STILL NOT CLEAR:

$$

\nabla_{\theta}J(\theta) = \mathbb{E}_{\tau \mathcal p_{\theta}(\tau)}[\sum_{t=0}^{t=T}\nabla_{\theta}log \pi_{\theta}(a_t|s_t)G_t]

$$

# Problem
The main problem that we want to solve is to maximize the expected return:

$$

\theta^{*} = \argmax_{\theta} \mathbb{E}_{\tau \mathcal p_{\theta}(\tau)} \sum_{t=0}^{t=T} log \pi_{\theta}(a_t|s_t) G_t

$$

# Observations
REINFORCE collects an episode. Calculates the returns. Then train the policy on the samples of this episode once (there is no batching the data).

It has to wait till the end of the episode to train the policy because it needs to first generate the returns and then train (It does not bootstrap).

The main difference between DQN and REINFORCE is that DQN using a replay buffer that uses previous samples to still train the current actiuon value function, therefore, it is an off-policy method. However, REINFORCE uses only the current data collected for the current episode to update the policy and it is not using the previous data to train the policy, therefore, it is an on-policy method.

The exploration part of the REINFORCE comes from the policy itself. The policy is generating probabilities for each action (in discrete action space). During the rollout, we sample from these probabilities. Sampling helps to try low probability actions. However, if the network starts to be biased towards one action, it becomes rare to try low probability ones.

REINFORCE compared with DQN needs more interactions with the environment (more episodes and more steps), since it is not using previous trajectories each time training the policy. In DQN, the replay buffer is a very important element as it helps to reuse the old samples along with the new collected samples.

I see a major issue with REINFORCE and that is exploration. REINFORCE is only dependant on the policy stochastic nature to produce random actions in the begining. Then the policy may become more confident about one particular set of actions. Then afterwards, mostly it will explore around that. However, in DQN, we have the $\epsilon$ greedy exploration that tries random actions even if the action value is confidant about something.

I observed that leaning rate for the policy training is very important. Reducing the learning rate prevents the network from overfitting very early.