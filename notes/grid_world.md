# Grid World Problem
A grid world is an N * N grid. There is an agent in a cell and the goal is to reach to the terminal cell. Following rules are applied in this world:
1. The bottom-left corner is the start state.
2. The top-right corner is the terminal state.
3. Every move gives a reward of -1.
4. The agent can move only up, right, left, and down.
5. For the boundary states, the agent stays at the same state if it tries to pass the boundary.

The world looks like this:

---------
|(0,3)|(1,3)|(2,3)|(3,3)|
---------
|(0,2)|(1,2)|(2,2)|(3,2)|
---------
|(0,1)|(1,1)|(2,1)|(3,1)|
---------
|(0,0)|(1,0)|(2,0)|(3,0)|
--------->x

Note: I'm using numpy array for the state-values. Therefore, to get the same x and y coordinates, I consider the following indexing:

(N-y,x)

where N is the size of the grid.

# Policy
The agent follows a random policy: at each state it takes a random action.

# Value Estimation
Bellman equation is being used to estimate the state-values.
