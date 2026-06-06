# Return
The return the sum of the rewards:

$$
G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T
$$

where T is the final time step of the episode.
# Discounted Return

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infinity}\gamma^k R_{t+k+1}
$$

where $\gamma$ is a parameter, $0\le \gamma \le 1$, called the discount rate.

Returns at successive time steps are related to each other:

$$
G_{t} = R_{t+1} + \gamma G_{t+1}
$$

Therefore, we can update the return at each time step, starting from the last time step assuming the last time step is $t=T$, therefore $G_{T+1}=0$:

**Pseudocode**

A collected episode following $\pi$: 

$$
S_0, A_0, R_1, S_1, A_1, R_2, \cdots, S_{T-1}, A_{T-1}, R_T
$$

1. G \leftarrow 0

2. Loop for each step of the episode, $t=T-1,T-2,\cdots,0$:

    $$
    G \leftarrow $\gamma$ G + R_{t+1}
    $$


# Expected Return

## Expected Return estimation using Monte Carlo
One way to estimate the expected return is to use the Monte Carlo methods by averaging the returns over different episodes:

**Pseudocode**

Loop for number of episodes:
    Generate an episode following $\pi$: 
    
    $$
    S_0, A_0, R_1, S_1, A_1, R_2, \cdots, S_{T-1}, A_{T-1}, R_T
    $$

    $G \leftarrow 0$

    Loop for each step of the episode, $t=T-1,T-2,\cdots,0$:

        $$
        G \leftarrow $\gamma$ G + R_{t+1}
        $$

        Append G to Returns($S_t$)


        $$
        V(S_t) \leftarrow average(Returns(S_t))
        $$
``` 

