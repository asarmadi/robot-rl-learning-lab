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
REINFORCE collects an episode. Calculates the returns. Then train the policy on the samples of this episode once.