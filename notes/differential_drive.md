# Differential Drive
It is a robot with two wheels of radius $r$ that are separated by a distance, let say $L$.

The goal in a this environment is to move the robot from point A to point B.

## Kinematics
The continuous-time kinematics of the system is as:

$$
\begin{aligned}
\dot{x} &= \frac{r}{2}(\omega_R+\omega_L)\cos{\theta}\\
\dot{y} &= \frac{r}{2}(\omega_R+\omega_L)\sin{\theta}\\
\dot{\theta} &= \frac{r}{L}(\omega_R+\omega_L)
\end{aligned}
$$

where $x$, $y$ are the location of the center point on the wheel axis in the global coordinate system, $\theta$ is the heading angle of the robot, $\omega_R$, $\omega_L$ are the angular speed of the right and left wheel.

## PPO
I designed the reward and tried to find the policy using PPO. One thing that I observed is that if the terminal reward is low, then the agent learns to hit the terminal state (e.g., hitting the boundaries) instead of going to the final target. The reason is that hitting a boundary results in a lower reward than continuing towards the target. Also, I observed that the reward for reaching to the goal at each time step should be comparable to the terminal reward to encourage the robot to move. In other words, we should penalize the robot for staying at the same location. Therefore, staying at the same location during the time limit should be more expensive than reaching the goal. Otherwise, the agent will learn to stay at the location for the whole time. To achieve this, I increased the coefficient for the distance to the goal.