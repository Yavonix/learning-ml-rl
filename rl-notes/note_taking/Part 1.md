## Chapter 1

**Tabular solution** methods: state/action spaces small enough for value function to be represented by tables. This is opposite to the **approximate methods**.

**Bandit problems**: only deal with a **single state**.

Techniques to solve finite Markov decision problems:
- Dynamic programming (require complete & accurate environment model)
- Monte Carlo methods (no model requirement, simple, not good for step-by-step incremental computation)
- Temporal difference learning (no model, fully incremental, more complex)

## Chapter 2 Multi-armed Bandits

### Intro

**Evaluative feedback**: how good was the action taken (used in RL)
**Instructive feedback**: what was the best action (used in supervised ML)

**Nonassociative**: means that there’s **no association between situations (states) and actions** — because there’s only _one_ situation.

### k-armed Bandit Problem
- k different actions to take
- After each action you receive a numerical reward (chosen from a stationary probability distribution)

“**k-armed**” just means there are _k_ slot machines (actions) to choose from
"**bandit**" comes from the fact that slot machines “steal” your money

Terminology:
- $\text{value}$: expected reward of an action.
- $A_t$: action selected at time step $t$. Is a _random variable_ depending on the agent policy.
- $R_t$: corresponding reward for $A_t$. Is a _random variable_ depending on the action and environment.
- $a$: arbitrary action.
- $q_*(a)$: value of the arbitrary action. Defined as the expected reward given a is selected.
$$q_*(a) = \mathbb{E}[R_t\mid A_t = a]$$
We do not know $q_*(a)$, so we define $Q_t(a)$ as the estimate. $Q_t(a)$ should be as close as possible to $q_*(a)$.

**Greedy** actions: actions with the greatest value.
Selecting **greedy** actions = **exploitation**.
Selecting **non-greedy** actions = **exploration**. (allows to update your estimate of the non-greedy action)

### Action-value Methods
#### Sample-average Method
Natural way to estimate action value:
$$Q_t(a)=\frac{\text{sum of rewards when a taken pior to t}}{\text{number of times a taken prior to t}} = 
\frac
{ \sum^{t-1}_{i=1} R_i \cdot \mathbb{1}_{A_i=a} }
{ \sum^{t-1}_{i=1}  \mathbb{1}_{A_i=a}  }
$$

#### Action Selection
- Greedy Action Selection
	Select the action with the highest value:
	$A_t\ \dot{=}\ \text{argmax}\ Q_t(a)$
- $\epsilon$-Greedy Methods 
	With probability $epsilon$ explore randomly.
	With probability $1-\epsilon$ take best action.
	The probability of selecting the optimal action converges to greater than $1-\epsilon$ (because you might accidentally select the most optimal action while exploring).

$\epsilon$-Greedy vs Greedy:
- High reward variance -> $\epsilon$-Greedy better
- Zero reward variance -> Greedy better

### $\alpha$-Filters

$$
\begin{align*}
Q_{n+1} &= \frac{1}{n} \sum_{i=1}^{n} R_i \\
&= \frac{1}{n} \left( R_n + \sum_{i=1}^{n-1} R_i \right) \\
&= \frac{1}{n} \left( R_n + (n-1) \frac{1}{n-1} \sum_{i=1}^{n-1} R_i \right) \\
&= \frac{1}{n} \left( R_n + (n-1) Q_n \right) \\
&= \frac{1}{n} \left( R_n + n Q_n - Q_n \right) \\
&= Q_n + \frac{1}{n} [ R_n - Q_n ],
\end{align*}
$$
Which forms:
$$\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize} \left[\text{Target} - \text{OldEstimate}\right]$$
Then for non-stationary problems, we introduce decay. This is termed **exponential recency weighted average**:
$$Q_{n+1} = Q_n + \alpha [ R_n - Q_n ]$$
### A Simple Bandit Algorithm

![[Part 1.png|center|3-4]]

### Tracking a non-stationary problem
Use alpha filters to decay older values.

Simulation where one agent uses online mean and the other uses alpha decay on a non-stationary 10-arm bandit problem:
![[Part 1 1.png|center|3-4]]

### Optimistic Initial Values
We can force exploration at the start by given Q high initial estimates. Overtime, these estimates will be dragged down after each arm has been explored at the start. Does not aid exploration in later steps.

![[Part 1 2.png|center|3-4]]

### 2.6 Unbiasing the Constant-Step-Size Trick
The standard alpha-filter has a bias toward the initial estimate:
$$
\begin{align*}
Q_{n+1} &= Q_n + \alpha [R_n - Q_n] \\
&= \alpha R_n + (1 - \alpha) Q_n \\
&= \alpha R_n + (1 - \alpha) [\alpha R_{n-1} + (1 - \alpha) Q_{n-1}] \\
&= \alpha R_n + (1 - \alpha)\alpha R_{n-1} + (1 - \alpha)^2 Q_{n-1} \\
&= \alpha R_n + (1 - \alpha)\alpha R_{n-1} + (1 - \alpha)^2 \alpha R_{n-2} + \\
&\quad \cdots + (1 - \alpha)^{n-1}\alpha R_1 + (1 - \alpha)^n Q_1 \\
&= (1 - \alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1 - \alpha)^{n - i} R_i
\end{align*}
$$
We can solve this by introducing an online mean $\bar{o}_n$ which is a trace of one that starts at 0:
$$\bar{o}_n = \bar{o}_{n-1} + \alpha (1-\bar{o}_{n-1}),\quad \text{for}\ n \ge0, \text{with}\ \bar{o}_0\ \dot{=}\ 0$$
And making the step size $\beta_n$:
$$\beta_n \ \dot{=} \ \frac{\alpha}{\bar{o}_n}$$
Which has the effect of removing the initial bias:
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
### 2.7 Upper-Confidence-Bound Action Selection
Instead of $A_t \ \dot{=} \ \underset{a}{\text{argmax}}(Q_t(a))$ we can also keep track of the uncertainty of the estimates:
$$
A_t \ \dot{=} \ \underset{a}{\text{argmax}}\left[
Q_t(a) 
+
c \cdot \sqrt{\frac{\ln(t)}{N_t(a)}}
\right]
$$
$c$ is a constant ($c>0$) that controls the degree of exploration.
$t$ is the current time step.
$\ln(t)$ makes sure that time step increases get smaller over time, but are unbounded.

This method does have some problems though:
- Problems with non-stationary problems.
- Suffers from combinatorial explosion in multi-state reinforcement learning.

![[Part 1 3.png|center|3-4]]

# TODO Make notes on last 2 parts

## Chapter 3 Finite Markov Decision Processes

### Extra
**Action value** function is the long term value of a state when choosing an action with policy $\pi$:
$$
q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]
$$
**State value** function $v_\pi$ is the same, but is the return when following a given policy $\pi$:
$$v_\pi = \mathbb{E}_\pi[G_t \mid S_t = s]$$
**Optimal state value** function is the maximum value function over all policies:
$$v_* = \max_\pi v_\pi (s)$$

**Optimal action value** function the maximum action value function over all policies:
$$ q_* (s,a) = \max_\pi q_\pi(s,a)$$
### Overview 
**Markov decision process trajectory**:
S is state, A is action, R is reward. Reward is given as part of the next state.
$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3,\ldots$$

MDP **dynamics** is defined by:
$$p(s', r \mid s, a) \doteq \Pr\{ S_t = s', R_t = r \mid S_{t-1} = s, A_{t-1} = a \},$$

Reward hypothesis:
> That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward)

Reward is given at a single timestep.
Return ($G_t$) is the expected cumulative reward.
$$G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \ldots + R_T$$
One simulation might be called an $\text{episode}$.
Each episode ends in a special state called the $\text{terminal state}$.
For our episodic tasks we define:
- $\mathcal{S}$ all states 
- $\mathcal{S}^+$ all states **excluding** the terminal state
**Time of termination**, $T$, is a random variable that normally varies from episode to episode.

Discounted return:
$$
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots 
=
\sum_{k=0}^\infty \gamma^k R_{t+k+1}
$$
If $\gamma=0$, the agent is “myopic”.

**Monte Carlo methods**: repeated random sampling to obtain numerical results

**Bellman Equation**:
The key here is the relationship between the value of a state and the values of its successor states.
$$
\begin{align}
v_\pi(s) 
&\doteq \mathbb{E}_\pi[G_t \mid S_t = s] \\
&= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \quad \text{(by (3.9))} \\
&= \sum_a \pi(a \mid s) \sum_{s'} \sum_r p(s', r \mid s, a) 
    \left[ r + \gamma \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] \right] \\
&= \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) 
    \left[ r + \gamma v_\pi(s') \right],
    \quad \forall s \in \mathcal{S}.
\end{align}
$$
You can find 3.9 in the textbook.