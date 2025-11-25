 2.1:
 - 2 Actions
 - $\epsilon = 0.5$
 - Probability that the greedy action is selected is $0.5 + 0.5\cdot \frac{1}{2} = 0.75$

2.2:
- The first time step we don't know if the $\epsilon$ occurred, but practically it makes no difference as our value estimate for each action is 0. So if we went down the greedy route it would still be random.
- The second time step was likely greedy, but we still dont know. It may have randomly chose a different action but more likely it arbitrarily picked from the higher value actions of 0.
- The third time is likely greedy.
- The fourth action must be $\epsilon$ because the value estimate of action 2 is -0.5 while action 3 and 4 have estimates of 0.
- The fifth action must be $\epsilon$ because the value estimate of action 2 is 0.33. while action 3 and 4 have estimates of 0.

2.3:
In the long run both $\epsilon=0.1$ and $\epsilon=0.01$ will attain close to true estimates of $q_*(a)$. However, $\epsilon=0.01$ will have a higher tendency to select the greedy option, therefore in the long run $\epsilon=0.01$ will achieve the best cumulative reward and highest probability of selecting the best action. For $\epsilon=0.01$ this is $0.99 + 0.01*1/10=0.991$ while $\epsilon=0.1$ this is $0.9 + 0.1*1/10=0.91$.

2.4:
?

2.5:
Done in jupyter

2.6:
We are using greedy with very high initial estimates. On each step the algorithm picks the action with the highest value. At the start the algorithm will go through each action and decay its estimate down from 5. The action in practise that has the highest value will decay the action value slower, therefore the algorithm might "cling" to this action for a little bit.

2.7:
$$
\begin{align*}
Q_{n+1} &= Q_n + \frac{\alpha}{\bar{o}_n}\left(R_n - Q_n\right) \\
&= \frac{\alpha}{\bar{o}_n}R_n + \left(1 - \frac{\alpha}{\bar{o}_n}\right)Q_n \\
&= \frac{\alpha}{\bar{o}_n}R_n + \left(1 - \frac{\alpha}{\bar{o}_n}\right)
    \left[\frac{\alpha}{\bar{o}_{n-1}}R_{n-1} + \left(1 - \frac{\alpha}{\bar{o}_{n-1}}\right)Q_{n-1}\right] \\
&= \frac{\alpha}{\bar{o}_n}R_n 
    + \left(1 - \frac{\alpha}{\bar{o}_n}\right)\frac{\alpha}{\bar{o}_{n-1}}R_{n-1} 
    + \left(1 - \frac{\alpha}{\bar{o}_n}\right)\left(1 - \frac{\alpha}{\bar{o}_{n-1}}\right)Q_{n-1} \\
&= \dots \\
&= \left[\prod_{i=1}^{n}\left(1 - \frac{\alpha}{\bar{o}_i}\right)\right] Q_1
   + \sum_{i=1}^{n}\frac{\alpha}{\bar{o}_i}
     \left[\prod_{k=i+1}^{n}\left(1 - \frac{\alpha}{\bar{o}_k}\right)\right]R_i
\end{align*}
$$
$$
\text{For } i = 1,\ \bar{o}_1 = \alpha(1 - 0) = \alpha
\implies \left(1 - \frac{\alpha}{\bar{o}_1}\right) = 1 - 1 = 0,
\text{ hence } Q_1 \text{ is not weighted.}
$$
2.8:
Reward increases on the 11th step as we are able to choose the action with the highest action-value estimate (each action has been tried once and all action-values have the same $c \cdot \sqrt{\frac{\ln(t)}{N_t(a)}}$). It falls on subsequent steps as it explores less optimal action due to the $c \cdot \sqrt{\frac{\ln(t)}{N_t(a)}}$ parameter.

## Chapter 3

3.1:
Ex 1: A lamp like the pixar lamp that is designed to illuminate your workspace. The state would be the current joint angles, joint velocities, and the gaze of a person. The action would be setting joint angles and the reward would be whether any paper the person is looking at is illuminated (pos reward) or how often the light is in their face (neg reward).
Ex 2: The space trader API game. State could be ships, their location, their configuration, and currently available trade routes. Actions could be sending a ship to a particular location or buying / selling goods. Reward could be credits.  
Ex 3: I know this is a classic example but I still wanted to do it: Balancing a pole on a linear actuator. The state is the pole angle, pole angular velocity, linear actuator position, linear actuator velocity. The reward is given continuously when the pole is vertical and is negative when the pole falls over. The action is linear actuator velocity commands.

3.2:
Given the state, the past is irrelevant for predicting the next step. Also:
- POMDPs: true state cannot be observed, only partially observed state is available. E.g., patient with sickness only reflected by noisy labs.
- Non-stationarity: transition/reward changes over calendar time. Can add time/task index to state but causes state to explode and adaptation non-trivial.
- Non-Markovian objectives: reward depends on a sequence of events. Once again state can be augmented to support this given sufficient memory.

3.3:
As you move the line "down" the agent must learn the "interactions" of each level down to achieve its goal making learning more difficult. Moreover, it would be irresponsible to use RL for instances where classic algorithms are more than sufficient to achieve optimal results. For instance, the RL algorithm could determine the optimal fuel ratio to adjust speed to get the car moving, but this would require also learning the dynamics of fuel injection. What would be better is that the RL algorithm output "speed" as an action and some highly refined, known classical system determines the correct fuel injection. I can see an argument for involving the additional complexity into the RL agent if the agent is meaningful able to make use of the additional complexity. For instance maybe the agent knows that using as much fuel as possible is part of the goal, where then it could adjust fuel injection to be least efficient to maximise the goal - something it could not have done if it could only output speed. There are fundamentals reasons for preferring one location over another, it is not free choice. 



3.5:
Instead of writing
$$\text{for all } s \in \mathcal{S}, a \in \mathcal{A}(s)$$
we could write
$$\text{for all } s \in \mathcal{S}^+, a \in \mathcal{A}(s)$$

3.6:
Episodic task the return is guaranteed to be $\ge -1$ while in the continuing task return may be smaller than $-1$.

3.7:
The return is 0 so the agent is never incentivised to "leave" the maze (in addition to it never discovering the exit).

3.8:

|     | Reward | Return |
| --- | ------ | ------ |
| G0  | -1     | 2      |
| G1  | 2      | 6      |
| G2  | 6      | 8      |
| G3  | 3      | 4      |
| G4  | 2      | 2      |
| G5  | 0      | 0      |

3.9:
$$
\gamma = 0.9, \quad R_1 = 2, \quad R_2 = R_3 = \dots = 7  
$$
$$
G_1 = 7 + 0.9(7) + 0.9^2(7) + \dots = 7 \sum_{k=0}^{\infty} 0.9^k = 7 \times \frac{1}{1-0.9} = 70  
$$
$$
G_0 = R_1 + \gamma G_1 = 2 + 0.9 \times 70 = 2 + 63 = 65  
$$

**3.10:**
**Required to prove:**
$$G_t = \sum^\infty_{k=0} \gamma^k=\frac{1}{1-\gamma}$$
$$
\begin{align}
G_t &= 1 + \gamma + \gamma^2 + \ldots \\
G_t \cdot \gamma &= \gamma + \gamma^2 + \gamma^3 + \ldots \\
G_t - \gamma G_t &= 1 \\
G_t(1 - \gamma) &= 1 \\
G_t &= \frac{1}{(1 - \gamma)}
\end{align}
$$

3.11:
$$
\mathbb{E}[R_{t+1} \mid S_t=s]=\sum_{a\in\mathcal{A}} \pi(a \mid s)\sum_{s'\in\mathcal{S}}   \sum_{r\in\mathcal{R}} r \cdot p(s',r\mid s,a)
$$
3.12:
$$
v_\pi (s)=\sum_{a\in\mathcal{A}} \pi(a\mid s) q_\pi(s, a)
$$
3.13:
$$
q_\pi(s, a) = \sum_{s'\in\mathcal{S}} \sum_{r\in\mathcal{R}} \left[
(\gamma \cdot v_\pi(s')+r) \cdot p(s',r\mid s,a) \right]
$$

3.14:
$$
\begin{align}
v_\pi(\text{center state}) &= 0.25\cdot(0+0.9\cdot 2.3)\\
&+ 0.25\cdot(0+0.9\cdot 0.7)\\
&+ 0.25\cdot(0+0.9\cdot 0.4)\\
&+ 0.25\cdot(0+0.9\cdot -0.4) \\
&=0.7
\end{align}
$$



399
434
