# Comparison between Algorithms
In this documnet, I'm going to summarize the comparison between various algorithms. To have a fair comparison between algorithms, I keep the following hyperparameters same for each environment: environment dynamics, step size, action limits, reward function, evaluation frequency, number of total episodes, random seed, neural network architecture, neural network initialization. 

|Features|SARSA|Q-learning|REINFORCE|DQN|A2C|PPO|DDPG|TD3|SAC|
|:------:|:---:|:--------:|:-------:|:-:|:-:|:-:|:--:|:-:|:-:|
|Learning Type|On-policy|Off-policy|On-policy|Off-policy|On-policy|On-policy|Off-policy|Off-policy|Off-policy|
|Action space|Discrete|Discrete|Both|Discrete|Both|Both|Continuous|Continuous|Continuous|
|Policy Type|$\epsilon$ greedy|$\epsilon$ greedy|Stochastic|$\epsilon$ greedy|Stochastic|Stochastic|Deterministic|Deterministic|Stochastic|
|Bootstraping|Yes|Yes|No|Yes|Yes|Yes|Yes|Yes|Yes|
|Replay Buffer|No|No|No|Yes|No|No|Yes|Yes|Yes|
|Number of networks|0|0|1|2|2|2|4|6|5|
|Target network|No|No|No|Yes|No|No|Yes|Yes|Yes|
|Exploration Mechanism|$\epsilon$ greedy|$\epsilon$ greedy|sampling|$\epsilon$ greedy|Sampling|Sampling/Entropy|Noise|Noise|Sampling/Entropy|
|Bias-Variance|Bootstrapping bias|Bootstrapping bias|High variance due to using return|Bootstrapping bias|Bootstrapping bias|Reduced bias by doing GAE|Bootstraping bias|Bootstrapping bias; twin critics reduce overestimation|Bootstrapping bias; twin critics reduce overestimation|
|Critic Type|Q|Q|N/A|Q|V|V|Q|Q|Q|
|Update Frequency|Every step|Every step|Every episode|Every step|Every step|Every rollout, for multiple epochs|Every step after warm-up|Critic Every step, actor less frequent|Every step after warm-up|
