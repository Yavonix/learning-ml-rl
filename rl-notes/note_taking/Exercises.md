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

3.4:
![[Exercises.png]]

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
3.15:
$$
v_c=\sum_{k=0}^{\infty} \gamma^k \cdot c=\frac{c}{1-\gamma}
$$
3.16:
Adding a positive constant generally encourages longer episodes.

3.17:
$$
\begin{align} q_{\pi}(s, a) &= \mathbb{E}_{\pi}[G_t \mid S_t = s, A_t = a] \\ &= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] \\ &= \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \mathbb{E}_{\pi}[G_{t+1} \mid S_{t+1} = s'] \right] \\ &= \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') q_{\pi}(s', a') \right] \end{align}
$$

3.18:
$$
\begin{aligned}
v_\pi(s) &= \mathbb{E}_{\pi} [ q_\pi(S_t, A_t) \mid S_t = s ] \\
&= \sum_{a \in \mathcal{A}} \pi(a|s) q_\pi(s, a)
\end{aligned}
$$

**3.19**:
$$
\begin{aligned}
q_\pi(s, a) &= \mathbb{E} [ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a ] \\
&= \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right]
\end{aligned}
$$

3.20:
It will be similar to $v_{putt}$ in terms of contour shape, but the contours will be far more spread out as we can use the driver to achieve more distance with the same number of swings.

3.21:
With $q_*(s, \text{putter})$ we are forced to start with a shortrange putter followed by the optimal choice. For this reason, compared to $q_*(s, \text{driver})$ we will yield better contours (equal to $v_*$) next to the flag pole (ie on the green) and worse contours far away.

3.22:
When $\gamma=0$, we care only about immediate reward, therefore optimal policy is $\pi_\text{left}$.
When $\gamma=0.5$, the policy $\pi_\text{left}$ has return:
$$G_t=1 + \sum_{k=0}^\infty 1\cdot 0.25 \cdot 0.25^{k} = 1.\overline{3}$$
while the policy $\pi_\text{right}$ has return:
$$
G_t = 0 + \sum_{k=0}^\infty 2 \cdot 0.5 \cdot 0.25^{k} = 1.\overline{3}
$$
So either policy is optimal.
When $\gamma=0.9$, the policy $\pi_\text{left}$ has return:
$$
G_t=1 + \sum_{k=0}^\infty 1\cdot 0.81 \cdot 0.81^{k} = 5.26
$$
while the policy $\pi_\text{right}$ has return:
$$
G_t = 0 + \sum_{k=0}^\infty 2 \cdot 0.9 \cdot 0.81^{k} = 9.47
$$
So policy $\pi_\text{right}$ is optimal.

3.23:
1. State: High ($h$)
$$\begin{aligned}
q_*(h, \text{wait}) &= r_{\text{wait}} + \gamma \max \left\{
    \begin{aligned}
    &q_*(h, \text{wait}), \\
    &q_*(h, \text{search})
    \end{aligned}
\right\} \\[1em]
q_*(h, \text{search}) &= r_{\text{search}} + \gamma \left(
    \alpha \max \left\{
        \begin{aligned}
        &q_*(h, \text{wait}), \\
        &q_*(h, \text{search})
        \end{aligned}
    \right\}
    + (1-\alpha) \max \left\{
        \begin{aligned}
        &q_*(l, \text{wait}), \\
        &q_*(l, \text{search}), \\
        &q_*(l, \text{recharge})
        \end{aligned}
    \right\}
\right)
\end{aligned}$$
2. State: Low ($l$)
$$\begin{aligned}
q_*(l, \text{wait}) &= r_{\text{wait}} + \gamma \max \left\{
    \begin{aligned}
    &q_*(l, \text{wait}), \\
    &q_*(l, \text{search}), \\
    &q_*(l, \text{recharge})
    \end{aligned}
\right\} \\[1em]
q_*(l, \text{recharge}) &= 0 + \gamma \max \left\{
    \begin{aligned}
    &q_*(h, \text{wait}), \\
    &q_*(h, \text{search})
    \end{aligned}
\right\} \\[1em]
q_*(l, \text{search}) &= \beta \left[ r_{\text{search}} + \gamma \max \left\{
    \begin{aligned}
    &q_*(l, \text{wait}), \\
    &q_*(l, \text{search}), \\
    &q_*(l, \text{recharge})
    \end{aligned}
\right\} \right] \\
&\quad + (1-\beta) \left[ -3 + \gamma \max \left\{
    \begin{aligned}
    &q_*(h, \text{wait}), \\
    &q_*(h, \text{search})
    \end{aligned}
\right\} \right]
\end{aligned}$$

3.24:
$$ \begin{aligned} v_*(A) &= 10 + \gamma^5(10) + \gamma^{10}(10) + \dots \\ &= \sum_{k=0}^{\infty} 10 \cdot (\gamma^5)^k \\ &= \frac{10}{1 - \gamma^5} \end{aligned} $$Given $\gamma = 0.9$: $$ v_*(A) = \frac{10}{1 - (0.9)^5} = \frac{10}{1 - 0.59049} = \frac{10}{0.40951} \approx 24.419 $$
3.25:
$$
v_*(s) = \max_{a \in \mathcal{A}(s)} q_*(s, a)
$$

3.26:
$$
q_*(s, a) = \sum_{s', r} p(s', r \mid s, a) [ r + \gamma v_*(s') ]
$$

3.27:
$$
\pi_*(a|s) = \begin{cases} \frac{1}{|A^*_s|} & \text{if } a \in A^*_s \\ 0 & \text{otherwise} \end{cases}
$$

3.28: (better notation)
$$ \pi_*(s) \doteq \operatorname*{argmax}_{a} \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_*(s') \right] $$

3.29:
Bellman Equations in terms of $r(s,a)$ and $p(s'|s,a)$

1. State-Value Function $v_\pi(s)$
$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \left[ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v_\pi(s') \right]
$$

2. Action-Value Function $q_\pi(s,a)$
$$
q_\pi(s,a) = r(s,a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) \sum_{a' \in \mathcal{A}} \pi(a'|s') q_\pi(s',a')
$$

3. Optimal State-Value $v_*(s)$
$$
v_*(s) = \max_{a \in \mathcal{A}} \left[ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v_*(s') \right]
$$

4. Optimal Action-Value $q_*(s,a)$
$$
q_*(s,a) = r(s,a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) \max_{a' \in \mathcal{A}} q_*(s',a')
$$

4.1:
Using:
$$
\begin{aligned}
q_\pi(s, a) &= \mathbb{E} [ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a ] \\
&= \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right]
\end{aligned}
$$
We have:
- $q_\pi(11, \text{down}) = -1$
- $q_\pi(7, \text{down}) = -1 + \gamma \cdot v_\pi = -1 + 1 \cdot -14 = -15$.


4.2:
Assuming dynamics of other cells unchanged.
First part:
$$
\begin{align}
v_\pi(15) &= \sum_a \pi(a\mid s) \sum_{s, r} p(s', r \mid s, a) \cdot \left[r + v_\pi(s') \right] \\
&= 0.25 \cdot \left[(-1 +-22) + (-1+-20) + (-1+-14) + (-1+v_\pi(15)) \right] \\
&= -20
\end{align}
$$
Second part:
$$
\begin{align}
v_\pi(13) &= \sum_a \pi(a\mid s) \sum_{s, r} p(s', r \mid s, a) \cdot \left[r + v_\pi(s') \right] \\
&= 0.25 \cdot \left[(-1 +-22) + (-1+-20) + (-1+-14) + (-1+v_\pi(15)) \right] \\
\\
v_\pi(15) &= \sum_a \pi(a\mid s) \sum_{s, r} p(s', r \mid s, a) \cdot \left[r + v_\pi(s') \right] \\
&= 0.25 \cdot \left[(-1 +-22) + (-1+ v_\pi(13)) + (-1+-14) + (-1+v_\pi(15)) \right] \\
\end{align}
$$
Solving yields: $v_\pi(15)=-20, v_\pi(13)=-20$

4.3:
For 4.3 and 4.4:
$$
\begin{align}
q_\pi(s,a) &= \mathbb{E}_\pi \left[ R_{t+1} + \gamma  q_\pi(S_{t+1}, A_{t+1}) \mid S_t=s, A_t=a \right] \\
&= \sum_{s',r} p(s',r \mid s, a) \cdot \left[r + \gamma \sum_{a'} \pi(a'\mid s') \cdot q_\pi(s', a') \right] 
\end{align}
$$
And for 4.5:
$$
q_{k+1}(s,a) = \sum_{s',r} p(s',r \mid s, a) \cdot \left[r + \gamma \sum_{a'} \pi(a'\mid s') \cdot q_k(s', a') \right] 
$$

4.4:
Either use a deterministic tie-breaking rule or better yet, **give priority to the current action**.

4.5:
$$
\begin{array}{l}
\textbf{1. Initialization} \\
\quad Q(s,a) \in \mathbb{R} \text{ and } \pi(s) \in \mathcal{A}(s) \text{ arbitrarily for all } s \in \mathcal{S}, a \in \mathcal{A} \\
\\
\textbf{2. Policy Evaluation} \\
\quad \textbf{Loop:} \\
\qquad \Delta \leftarrow 0 \\
\qquad \textbf{Loop for each } s \in \mathcal{S}: \\
\qquad \qquad \textbf{Loop for each } a \in \mathcal{A}: \\
\quad \qquad \qquad q \leftarrow Q(s,a) \\
\quad \qquad \qquad Q(s,a) \leftarrow \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma Q(s',\pi(s')) \right] \\
\quad \qquad \qquad \Delta \leftarrow \max(\Delta, |q - Q(s)|) \\
\quad \textbf{until } \Delta < \theta \text{ (a small positive number determining the accuracy of estimation)} \\
\\
\textbf{3. Policy Improvement} \\
\quad \textit{policy-stable} \leftarrow \textit{true} \\
\quad \textbf{For each } s \in \mathcal{S}: \\
\qquad \textit{old-action} \leftarrow \pi(s) \\
\qquad \pi(s) \leftarrow \underset{a}{\mathrm{argmax}}\ Q(s,a) \\
\qquad \textbf{If } \textit{old-action} \neq \pi(s), \textbf{ then } \textit{policy-stable} \leftarrow \textit{false} \\
\quad \textbf{If } \textit{policy-stable}, \textbf{ then stop and return } V \approx v_* \text{ and } \pi \approx \pi_*; \textbf{ else go to 2}
\end{array}
$$

4.6:
In 3 we would need some mechanism of assigning probabilities of selecting each action as opposed to taking a simple argmax. This might mean:
- Greedy action $a^*=1-\epsilon+\frac{\epsilon}{|\mathcal{A}(s)|}$
- Other actions $= \frac{\epsilon}{|\mathcal{A}(s)|}$

In 2 when performing the value update we would need to consider each action, this would be a weighted average against the probabilities of each action being selected over the $r + \gamma V(s')$. I.e., $V(s)\leftarrow \sum_a \pi(a\mid s) \sum_{s',r} p(s',r\mid s, a)[r+\gamma V(s')]$ 


In 1 the $\pi(s)$ assignment would need to obey the $\epsilon\text{-soft}$ requirement.

4.7:
```python
import math
import numpy as np
from scipy.stats import poisson


np.set_printoptions(precision=1)


print("Precomputing state transition probabilities...")

max_cars = 21
poisson_tail = 50

transition_A = np.zeros((max_cars,max_cars))
transition_B = np.zeros((max_cars,max_cars))
expected_reward_A = np.zeros((max_cars)) # map starting number of cars to expected reward
expected_reward_B = np.zeros((max_cars)) # map starting number of cars to expected reward

pmf_map = {
    3: [poisson.pmf(i, 3) for i in range(poisson_tail)],
    4: [poisson.pmf(i, 4) for i in range(poisson_tail)],
    2: [poisson.pmf(i, 2) for i in range(poisson_tail)]
}

for starting_cars in range(0, max_cars):
    for returned_cars in range(0, poisson_tail):
        for rented_cars in range(0, poisson_tail):
            actual_rented = min(starting_cars, rented_cars)
            final_cars = min(starting_cars + returned_cars - actual_rented, 20)

            prob_A = pmf_map[3][rented_cars] * pmf_map[3][returned_cars]
            prob_B = pmf_map[4][rented_cars] * pmf_map[2][returned_cars]

            transition_A[starting_cars, final_cars] += prob_A
            transition_B[starting_cars, final_cars] += prob_B

            expected_reward_A[starting_cars] += prob_A * 10 * actual_rented
            expected_reward_B[starting_cars] += prob_B * 10 * actual_rented

print("Finshed...")

def apply_action(action, state):
    return tuple(np.clip((state[0] - action, state[1] + action), 0, 20))

def compute_reward(values, action, state, gamma):
    reward = 0
    if action >= 0: # Move A -> B (limited by cars at A)
        actual_move = min(action, state[0])
        if actual_move > 0:
            reward += 2 # One of Jack's employees will shuttle car for free.
    else: # Move B -> A (limited by cars at B)
        actual_move = -min(abs(action), state[1])
    reward += -2*abs(actual_move)
    next_direct_state = apply_action(actual_move, state)

    if next_direct_state[0] > 10: reward -= 4
    if next_direct_state[1] > 10: reward -= 4

    reward += expected_reward_A[next_direct_state[0]] + expected_reward_B[next_direct_state[1]]

    return reward + gamma * np.sum(np.outer(transition_A[next_direct_state[0], :], transition_B[next_direct_state[1], :]) * values)

def eval_policy(values, policy, states, gamma, threshold):
    while True:

        delta = 0
        for state in states:
            old_value = values[state]
            action = policy[state]
            
            values[state] = compute_reward(values, action, state, gamma)

            delta = max(delta, abs(old_value-values[state]))

        if delta < threshold:
            return values
        
def improve_policy(values, policy, states, gamma):
    policy_stable = True
    for state in states:
        old_action = policy[state]

        min_action = -min(5, state[1]) # Can't move more from A than A has (max move is 5)
        max_action = min(5, state[0]) # Only iterate over valid actions

        best = (0,-math.inf)
        for action in range(min_action, max_action + 1):
            predicted_reward = compute_reward(values, action, state, gamma)
            best = max(best, (action, predicted_reward), key=lambda x: x[1])

        policy[state] = best[0]
        if old_action != policy[state]: policy_stable = False
    
    return policy, policy_stable

states = [(i//max_cars, i%max_cars) for i in range(max_cars*max_cars)]
policy = np.zeros((max_cars,max_cars), dtype=int)
values = np.zeros((max_cars,max_cars))

gamma = 0.9

stable = False

while not stable:
    values = eval_policy(values, policy, states, gamma, 0.001)
    policy, stable = improve_policy(values, policy, states, gamma)

print(policy)

```

4.8:
It's mostly about protecting downside. At $50 we stake it all and have a 0.4 chance of winning. At $51 we can stake $1, maybe we win and we are in a better position, or we lose and we still have a second chance to win it all at $50.

4.9:
```python
## Gambler's Problem

import math
import numpy as np
import plotly.express as px

states = np.arange(1,100) # [1, 99]
values = np.zeros((101)) # [0, ..., 1]
values[100] = 1
threshold = 1e-30
p_heads = 0.1

def compute_reward(state, action, values, p_heads) -> float:
    reward = 0
    ## assume lost stake
    new_amount = state-action
    if (new_amount != 0): reward += (1-p_heads) * values[new_amount] 
    ## assume win stake
    new_amount = state+action
    reward += p_heads * values[new_amount]

    return reward


while True:
    delta = 0
    for state in states:
        old_value = values[state]

        best = (0,-math.inf)
        for action in range(0, min(state, 100-state)+1):
            reward = compute_reward(state, action, values, p_heads)
            best = max((action, reward), best, key=lambda x: x[1])
        values[state] = best[1]

        delta = max(abs(values[state] - old_value), delta)

    if delta < threshold:
        break

## Extract policy

policy = np.zeros_like(values)

for state in states:
    best = (0,-math.inf)
    for action in range(1, min(state, 100-state)+1):
        reward = compute_reward(state, action, values, p_heads)
        reward = round(reward, 10)
        best = max(best, (action, reward), key=lambda x: x[1])
    policy[state] = best[0]

fig = px.line(x=np.arange(0, 101), y=values)
fig.add_bar(x=np.arange(0, 101), y=policy, name='policy', opacity=0.6, yaxis='y2')

fig.update_layout(
    yaxis=dict(title='value'),
    yaxis2=dict(title='policy', overlaying='y', side='right', showgrid=False),
    margin=dict(t=60)
)

fig.show()
print(values)
print(policy)
```
![[Exercises 1.png|center]]

4.10:
$$
q_{k+1}(s,a) \doteq \sum_{s',r} p(s',r \mid s,a) \cdot [r + \gamma \max_{a'} q_k(s', a')]
$$
5.1:
The evaluated policy only sticks on the last two rows. With other rows sticking has a high chance of ending the game in a loss. The value function drops off for the whole last row on the left because the dealer has a higher chance of winning, as the dealer as the ace, if when the dealer hits they go over, they can simply count their ace as 1. The frontmost values are higher in the upper diagram because the usable ace is an insurance policy against going over as it can be counted as 1.

5.2:
We never return to the same state, therefore every-visit MC will be the same.

5.3:
Start with action not state.

5.4:
This is just referring to using the online mean formula. Returns(s,a) would be a pair of real numbers for all s and a. The update would then be performed as $w_{k+1} \leftarrow w_k + \frac{(\text{new\_value} - w_k)}{n}$.  

5.5:
First visit value = 10. Every visit value = $\frac{10+9+8+7+6+5+4+3+2+1}{10}=5.5$.