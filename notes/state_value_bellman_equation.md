# State-value Bellman Equation
The Bellman equation for the state-value function defines a relationship between the value of a state and the value of its possible successor states. 

We start by the definition of the state-value function:

$$
\begin{aligned}
v_{\pi}(s) &\triangleq \mathbb{E}_{\pi}[G_t \mid S_t = s] \\
&= \mathbb{E}_{\pi}[R_{t+1}+\gamma G_{t+1} \mid S_t=s]\\
&= \sum_{a}\pi(a\mid s)\sum_{s'} \sum_r p(s', r\mid s, a)[r+\gamma \mathbb{E}_{\pi}[G_{t+1}\mid S_{t+1}=s']]\\
&= \sum_{a}\pi(a\mid s)\sum_{s', r}p(s', r \mid s, a)[r+\gamma v_{\pi}(s')], for all s \in \mathcal{S}
\end{aligned}
$$
