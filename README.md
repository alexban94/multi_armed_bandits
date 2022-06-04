# Multi-armed Bandits
C++ implementation of the multi-armed bandit problem and the 10-armed testbed as described in Reinforcement Learning: An Introduction (Sutton and Barto).

The bandit problem involves selecting from $k$ actions over a number of iterations, with the goal to maximize the reward recieved. The problem is set up as such:

* The expected reward given that an action $a$ is selected is denoted $q_\ast(a)$ and is sampled from a normal distribution of $\mu = 0$ and $\sigma^2=1$ for each $a$. 
* Given these, as the problem is solved, the reward $R_t$ for an action $A_t$ selected at timestep $t$ is sampled from a normal distribution of $\mu = q_\ast(A_t)$ and $\sigma^2=1$, denoted by the function ```bandit(a)```.

The pseudocode for the algorithm is as follows:

```
INITIALIZATION
for a = 1 to k:
  Q(a) = 0
  N(a) = 0
  
RUN
for t = 1 to max_iterations:
  Sample either:
    A = argmax Q(a)       with probability 1 - ε
    A = a random action   with probability ε
  R = bandit(A)
  N(A) = N(A) + 1
  Q(A) = Q(A) + 1/N(A) * (R(A) - Q(A))
```

This code incorporates the sample average, ```Q(a)``` an estimate of the expected reward, to guide the choice of action; for each action, ```Q(a)``` is the sum of the rewards obtained by the action, divided by the number of times it has been selected. At each timestep it is updated incrementally to avoid unnecessary computation and memory allocation. 

If $ε > 0$, then random actions are allowed with probability ε, permitting some exploration and the potential for a better action to be found.

For a given set of parameters, _max_iterations_, $k$ and $ε$, randomly generated bandit problems are evaluated over 2000 runs using the average reward obtained and the average accuracy of selecting the optimal action, at each timestep $t$. Please note that the optimal action for a bandit problem is defined as $argmax$   $q_\ast(a)$.
