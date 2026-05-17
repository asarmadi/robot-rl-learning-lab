# Model-Based
## Definition
The RL algorithms that have access to a model of the environment. A model is a function that predicts the state transitions and rewards.
## Advantages
1. It allows the agent to plan by thinking ahead, seeing what would happen for possible choices, and explicitly deciding between its options.
2. Sample Efficiency.
## Limitations
1. The agent has to learn the model purely from experience.
2. The agnet might exploit the learned model and not behaving as well on other environments.
3. Model learning is fundamentally hard.

# Model-Free
## Definition
The RL algorithms that do not use a model.
## Advantages
1. Much easier to implement and tune.
## Limitations
1. They are not sample efficient.