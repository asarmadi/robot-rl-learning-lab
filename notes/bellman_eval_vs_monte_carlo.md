# Bellman Evaluation vs Monte Carlo Value Estimation
Monte Carlo samples the state space to get an estimation of the state values. While Bellman evaluation needs to visit every state in the state space to update the valu function. However, to get a good estimate in Monte Carlo, many episode should be run.

Monte Carlo is updating the state values in the end of the episode. If a state was not seen by the agent during the episode. It won't be updated. In Fig. ![Alt Text](logs/mc_vs_bellman_grid/state_values/0.png), you can see that the (3,0) state value remain zero as the agent did not visit that state. This shows that Monte Carlo needs to be run for many episodes to reach bellman performance.

On the other hand, Bellman needs to sweep all the states in the state space.