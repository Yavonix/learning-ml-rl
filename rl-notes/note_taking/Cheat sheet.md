# Ch 3 Finite Markov Decision Processes
## Dynamics of the MDP
MDP **dynamics** is defined by: (3.2)
$$p(s', r \mid s, a) \doteq \Pr\{ S_t = s', R_t = r \mid S_{t-1} = s, A_{t-1} = a \} \tag{3.2}$$
Where:
$$
\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r \mid s, a) = 1, \text{ for all } s \in \mathcal{S}, a \in \mathcal{A}(s). \tag{3.3}
$$
From this we can derive:
* **State-transition** probabilities: (3.4)
  $p(s'|s, a) \doteq \Pr\{S_t=s' | S_{t-1}=s, A_{t-1}=a\} = \sum_{r \in \mathcal{R}} p(s', r|s, a).$
* **Expected-reward** for state-actions: (3.5)
  $r(s, a) \doteq \mathbb{E}[R_t | S_{t-1}=s, A_{t-1}=a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r|s, a),$
* **Expected-reward** for state-action-next-state: (3.6)
  $r(s, a, s') \doteq \mathbb{E}[R_t | S_{t-1}=s, A_{t-1}=a, S_t=s'] = \sum_{r \in \mathcal{R}} r \frac{p(s', r|s, a)}{p(s'|s, a)}.$
## Return and Reward
If the sequence of rewards received after time step t is denoted
$R_{t+1}, R_{t+2}, R_{t+3}, \dots$ then **return** over an **episode** is defined by: (3.7)
$$G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \dots+ R_T \tag{3.7}$$
For an **episodic task** to end we need a special **terminal state**, we update our set definitions as:
- $\mathcal{S}$ all states
- $\mathcal{S}^+$ all states including the **terminal state**
As opposed to **continuing task**s which are indefinite. Regardless of task type we should use **discounted return**: (3.8)
$$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots 
=
\sum_{k=0}^\infty \gamma^k R_{t+k+1} \tag{3.8}$$
Where current return is dependent on future return: (3.9)
$$
\begin{align} G_t &\doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots \\ &= R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \cdots) \\ &= R_{t+1} + \gamma G_{t+1} \tag{3.9} \end{align}
$$

## Unified Notation for Episodic and Continuing Tasks
$S_{t,i}$ is state at time $t$ of episode $i$.

We also have an alternate definition for 3.8: (3.11)
$$
G_t \doteq \sum_{k=t+1}^T \gamma^{k-t-1} R_{k} \tag{3.11}
$$
## Policies and Value Functions
A **policy** maps *states* to probabilities of selecting an *action*. E.g., $\pi(a\mid s)$.

**State-value function**: expected return for a state: (3.12)
$$
v_{\pi}(s) \doteq \mathbb{E}_{\pi}[G_t \mid S_t = s] = \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \biggm| S_t = s\right], \text{ for all } s \in \mathcal{S} \tag{3.12}
$$
**Action-value function**: expected return for an action: (3.13)
$$
q_{\pi}(s, a) \doteq \mathbb{E}_{\pi}[G_t \mid S_t = s, A_t = a] = \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \biggm| S_t = s, A_t = a\right] \tag{3.13}
$$
3.12 can be written recursively to represent the *Bellman equation* for $v_\pi$: (3.14)
$$
\begin{align}
v_{\pi}(s) &\doteq \mathbb{E}_{\pi}[G_t \mid S_t = s] \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \qquad \text{(by (3.9))} \\
&= \sum_{a} \pi(a|s) \sum_{s'} \sum_{r} p(s', r \mid s, a) \left[ r + \gamma \mathbb{E}_{\pi}[G_{t+1} \mid S_{t+1} = s'] \right] \\
&= \sum_{a} \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_{\pi}(s') \right], \quad \text{for all } s \in \mathcal{S}, \tag{3.14}
\end{align}
$$
Backup diagram:
![[Cheat sheet.png|center]]

3.13 can be written recursively to represent the *Bellman equation* for $q_\pi$:
$$
\begin{align}
q_{\pi}(s, a) &= \mathbb{E}_{\pi}[G_t \mid S_t = s, A_t = a] \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] \\
&= \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \mathbb{E}_{\pi}[G_{t+1} \mid S_{t+1} = s'] \right] \\
&= \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') q_{\pi}(s', a') \right]
\end{align}
$$

Backup diagram:
![[Cheat sheet 1.png|center]]

## Optimal Policies and Optimal Value Functions
An *optimal policy* ($\pi_*$) consists of an *optimal state-value function* ($v_*(s)$) as defined by: (3.15)
$$
v_*(s) \doteq \max_\pi v_\pi(s) \tag{3.15}
$$
and an *optimal action-value function* ($q_*(s,a)$) as defined by: (3.16)
$$
q_*(s,a) \doteq \max_\pi q_\pi(s,a) \tag{3.16}
$$
for all $s \in \mathcal{S}$, $a \in \mathcal{A}$. Also $q_*(s,a)$ can be written as: (3.17)
$$
q_*(s,a)=\mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1}) \mid S_t=s, A_t=a] \tag{3.17}
$$

The Bellman equation for $v_*$ is termed the *Bellman optimality equation*: (two forms: 3.18, 3.19)
$$
\begin{align}
v_*(s) &= \max_{a \in \mathcal{A}(s)} q_{\pi_*}(s, a) \\
&= \max_{a} \mathbb{E}_{\pi_*} [ G_t \mid S_t = s, A_t = a ] \\
&= \max_{a} \mathbb{E}_{\pi_*} [ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a ] & \qquad \text{(from (3.9))} \\
&= \max_{a} \mathbb{E} [ R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a ] \tag{3.18} \\
&= \max_{a} \sum_{s', r} p(s', r \mid s, a) [ r + \gamma v_*(s') ]. \tag{3.19}
\end{align}
$$
The *Bellman optimality equation* for $q_*$ is: (3.20)
$$
\begin{align}
q_*(s, a) &= \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t = s, A_t = a \right] \\
&= \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \max_{a'} q_*(s', a') \right]. \tag{3.20}
\end{align}
$$
Backup diagrams:
![[Cheat sheet 2.png|center]]

# Ch 4 Dynamic Programming
## Refreshing the Bellman optimality equations:
$$
\begin{align} v_*(s) &= \max_{a} \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t=s, A_t=a] \\ &= \max_{a} \sum_{s',r} p(s',r|s,a)[r + \gamma v_*(s')], \text{ or} \tag{4.1} \\
q_*(s,a) &= \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t=s, A_t=a\right] \\ &= \sum_{s',r} p(s',r|s,a)\left[r + \gamma \max_{a'} q_*(s', a')\right], \tag{4.2}
\end{align}
$$


## Policy Evaluation
*Policy evaluation* is the act of computing $v_\pi$ for arbitrary policy $\pi$. Also known as the *prediction problem*.

Recall the *Bellman equation* for $v_\pi$: (3.14) (4.4)
$$
\begin{align}
v_{\pi}(s) &\doteq \mathbb{E}_{\pi}[G_t \mid S_t = s] \nonumber \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \qquad \text{(from (3.9))} \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s] \tag{4.3} \\
&= \sum_{a} \pi(a|s) \sum_{s', r} p(s', r|s, a) \left[ r + \gamma v_{\pi}(s') \right], \tag{4.4}
\end{align}
$$
Existence and uniqueness of $v_\pi$ predicated on $\gamma < 1$ or eventual termination. If environment dynamics known, then (4.4) is a system of $|\mathcal{S}|$ simultaneous equations with $|\mathcal{S}|$ unknowns ($v_\pi(s), s \in \mathcal{S}$). For solving we use iterative methods.

*Iterative Policy Evaluation*
We are computing the value for each state when a given policy is used.

For approximate value functions $v_0, v_1, v_2, \dots$ mapping $\mathcal{S}^+ \to \mathbb{R}$, $v_0$ may be chosen arbitrarily except the terminal state must be given 0. Each successive approximation is obtained by the Bellman equation for $v_\pi$ (4.4): (4.5)
$$
\begin{align} v_{k+1}(s) &\doteq \mathbb{E}_{\pi}[R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t=s] \nonumber \\ &= \sum_{a} \pi(a|s) \sum_{s',r} p(s',r|s,a) \left[r + \gamma v_k(s')\right], \tag{4.5} \end{align}
$$
for all $s\in \mathcal{S}$. Substituting $v_\pi$ for $v_k$ produces $v_{k+1} = v_\pi$ demonstrating the fixed point.

Instead of using separate $v_0, v_1, v_2, \dots$ we can instead use one $v$ and perform in-place updates. This usually converges faster.

*Value Iteration Algorithm*:
$$
\begin{array}{l}
\textbf{Input: } \pi, \text{ the policy to be evaluated} \\
\textbf{Algorithm parameter: } \text{a small threshold } \theta > 0 \text{ determining accuracy of estimation} \\
\textbf{Initialize } V(s), \text{ for all } s \in \mathcal{S}^+, \text{ arbitrarily except that } V(\mathrm{terminal}) = 0 \\
\\
\textbf{Loop:} \\
\quad \Delta \leftarrow 0 \\
\quad \textbf{Loop for each } s \in \mathcal{S}: \\
\qquad v \leftarrow V(s) \\
\qquad V(s) \leftarrow \sum_{a} \pi(a|s) \sum_{s', r} p(s', r|s, a) \left[ r + \gamma V(s') \right] \\
\qquad \Delta \leftarrow \max(\Delta, |v - V(s)|) \\
\textbf{until } \Delta < \theta
\end{array}
$$

*Policy Improvement Theorem*
Given a policy $\pi'$, such that: (4.7)
$$
q_\pi(s, \pi'(s)) \ge v_\pi(s) \tag{4.7}
$$
Then by the policy improvement theorem: (4.8)
$$
v_{\pi'}(s) \ge v_\pi(s) \tag{4.8}
$$
Proof on page 78.

The natural choice for $\pi'$ is the greedy policy given by:
$$
\begin{align}
\pi'(s) &\doteq \underset{a}{\mathrm{argmax}} \, q_{\pi}(s, a) \nonumber \\
&= \underset{a}{\mathrm{argmax}} \, \mathbb{E}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t=s, A_t=a] \tag{4.9} \\
&= \underset{a}{\mathrm{argmax}} \sum_{s', r} p(s', r|s, a) \left[ r + \gamma v_{\pi}(s') \right], \nonumber
\end{align}
$$
The idea of constructing a new policy by making it greedy with respect to the value function of the original policy is called *policy improvement*.

Supposing the new policy $\pi'$ is as good as but not better than the old policy $\pi$, then $v_\pi=v_{\pi'}$ and from (4.9) it follows that:
$$
\begin{align}
v_{\pi'}(s) &= \max_{a} \mathbb{E}[R_{t+1} + \gamma v_{\pi'}(S_{t+1}) \mid S_t=s, A_t=a] \\
&= \max_{a} \sum_{s', r} p(s', r|s, a) \left[ r + \gamma v_{\pi'}(s') \right].
\end{align}
$$
Which is the same as the Bellman optimality equation (4.1), therefore $v_{\pi'}$ must be $v_*$ and, $\pi$ and $\pi'$ are optimal policies.

## Policy Iteration
Once we have formed an improved policy $\pi_1$ from our original policy $\pi_0$ that is greedy w.r.t $v_{\pi_0}$, we may use $\pi_1$ as our new policy to evaluate:
$$
\pi_0 \xrightarrow{\text{E}} v_{\pi_0} \xrightarrow{\text{I}} \pi_1 \xrightarrow{\text{E}} v_{\pi_1} \xrightarrow{\text{I}} \pi_2 \xrightarrow{\text{E}} \dots \xrightarrow{\text{I}} \pi_* \xrightarrow{\text{E}} v_*
$$
Where $\xrightarrow{\text{E}}$ denotes *evaluation* and $\xrightarrow{\text{I}}$ denotes *improvement*.

*Iterative Policy Evaluation Algorithm*:
Note how we use the same value function in each policy evaluation (increases convergence speed).
```tabs
tab: State-Value Iterative Policy Evaluation
$$
\begin{array}{l}
\textbf{1. Initialization} \\
\quad V(s) \in \mathbb{R} \text{ and } \pi(s) \in \mathcal{A}(s) \text{ arbitrarily for all } s \in \mathcal{S} \\
\\
\textbf{2. Policy Evaluation} \\
\quad \textbf{Loop:} \\
\qquad \Delta \leftarrow 0 \\
\qquad \textbf{Loop for each } s \in \mathcal{S}: \\
\quad \qquad v \leftarrow V(s) \\
\quad \qquad V(s) \leftarrow \sum_{s', r} p(s', r \mid s, \pi(s)) \left[ r + \gamma V(s') \right] \\
\quad \qquad \Delta \leftarrow \max(\Delta, |v - V(s)|) \\
\quad \textbf{until } \Delta < \theta \text{ (a small positive number determining the accuracy of estimation)} \\
\\
\textbf{3. Policy Improvement} \\
\quad \textit{policy-stable} \leftarrow \textit{true} \\
\quad \textbf{For each } s \in \mathcal{S}: \\
\qquad \textit{old-action} \leftarrow \pi(s) \\
\qquad \pi(s) \leftarrow \underset{a}{\mathrm{argmax}} \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V(s') \right] \\
\qquad \textbf{If } \textit{old-action} \neq \pi(s), \textbf{ then } \textit{policy-stable} \leftarrow \textit{false} \\
\quad \textbf{If } \textit{policy-stable}, \textbf{ then stop and return } V \approx v_* \text{ and } \pi \approx \pi_*; \textbf{ else go to 2}
\end{array}
$$


tab: Action-Value Iterative Policy Evaluation
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
\quad \qquad \qquad \Delta \leftarrow \max(\Delta, |q - Q(s,a)|) \\
\quad \textbf{until } \Delta < \theta \text{ (a small positive number determining the accuracy of estimation)} \\
\\
\textbf{3. Policy Improvement} \\
\quad \textit{policy-stable} \leftarrow \textit{true} \\
\quad \textbf{For each } s \in \mathcal{S}: \\
\qquad \textit{old-action} \leftarrow \pi(s) \\
\qquad \pi(s) \leftarrow \underset{a}{\mathrm{argmax}}\ Q(s,a) \\
\qquad \textbf{If } \textit{old-action} \neq \pi(s), \textbf{ then } \textit{policy-stable} \leftarrow \textit{false} \\
\quad \textbf{If } \textit{policy-stable}, \textbf{ then stop and return } Q \approx q_* \text{ and } \pi \approx \pi_*; \textbf{ else go to 2}
\end{array}
$$
```
## Value Iteration
Generally, beyond the first few iterations of policy evaluation, further iterations have no impact on the corresponding greedy policy. Therefore we could combine policy eval and policy improvement into one step.

This algorithm is called *value iteration*: (4.10)
$$
\begin{align}
v_{k+1}(s) &\doteq \max_{a} \mathbb{E}[R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t=s, A_t=a] \\
&= \max_{a} \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k(s')], \tag{4.10}
\end{align}
$$
Equivalent action-value algorithm:
$$
q_{k+1}(s,a) \doteq \sum_{s',r} p(s',r \mid s,a) \cdot [r + \gamma \max_{a'} q_k(s', a')]
$$

*Value iteration algorithm*:
$$
\begin{array}{l}
\textbf{Value Iteration, for estimating } \pi \approx \pi_* \\
\textbf{Algorithm parameter: } \text{a small threshold } \theta > 0 \text{ determining accuracy of estimation} \\
\textbf{Initialize } V(s), \text{ for all } s \in \mathcal{S}^+, \text{ arbitrarily except that } V(\mathrm{terminal}) = 0 \\
\\
\textbf{Loop:} \\
\quad \Delta \leftarrow 0 \\
\quad \textbf{Loop for each } s \in \mathcal{S}: \\
\qquad v \leftarrow V(s) \\
\qquad V(s) \leftarrow \max_a \sum_{s', r} p(s', r|s, a) \left[ r + \gamma V(s') \right] \\
\qquad \Delta \leftarrow \max(\Delta, |v - V(s)|) \\
\textbf{until } \Delta < \theta \\
\\
\text{Output a deterministic policy, } \pi \approx \pi_*, \text{ such that} \\
\quad \pi(s) = \operatorname*{argmax}_a \sum_{s', r} p(s', r|s, a) \left[ r + \gamma V(s') \right]
\end{array}
$$

We can interpose multiple policy evaluation sweeps between each policy improvement sweep to increase the speed of convergence. This takes the shape of replacing the value iteration sweep $V(s) \leftarrow \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]$ with  $V(s) \leftarrow \sum_{s',r} p(s',r|s,\pi(s))[r + \gamma V(s')]$. (Max is the only difference).

## Asynchronous Dynamic Programming
Asynchronous DP algorithms can update the values of states in any order whatsoever, using whatever values of other states happen to be available.
- Some states may be updated several times before others
- For convergence all states must be updated.

There exists orderings where convergence does not occur (apparently easy to avoid these.. idk)

We can also:
- Select states such that applying updates improves the algos rate of progress.
- Order updates to optimise information propagation.
- Avoid updating unnecessary state (explored in 8)
- There's also a bit on intermixing with real-time interaction in the chapter.

## Generalised Policy Iteration
GPI refers to the general idea of letting policy-evaluation and policy-improvement interact. To recap:
- Value iteration: $V(s) \leftarrow \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]$
- Policy evaluation: $V(s) \leftarrow \sum_{s',r} p(s',r|s,\pi(s))[r + \gamma V(s')]$
- Policy improvement: $\pi(s) \doteq \underset{a}{\mathrm{argmax}} \sum_{s', r} p(s', r|s, a) \left[ r + \gamma v_{\pi}(s') \right]$



# Ch 5 Monte Carlo Methods
Refers to methods that average complete returns, while Temporal Difference learning can learn from partial returns.

Unlike Dynamic Programming approaches, Monte Carlo methods require only *experience*, not complete knowledge of the environment.

Learning from *simulated* experience is also possible. Where compared to DP, this does not require complete probability distributions of all possible transitions.

We define Monte Carlo methods only for episodic tasks.

$$
\begin{array}{l} \textbf{First-visit MC prediction, for estimating } V \approx v_{\pi} \\ \\ \textbf{Input: } \text{a policy } \pi \text{ to be evaluated} \\ \textbf{Initialize:} \\ \quad V(s) \in \mathbb{R}, \text{ arbitrarily, for all } s \in \mathcal{S} \\ \quad Returns(s) \leftarrow \text{an empty list, for all } s \in \mathcal{S} \\ \\ \textbf{Loop forever } (\text{for each episode}): \\ \quad \text{Generate an episode following } \pi: S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T \\ \quad G \leftarrow 0 \\ \quad \textbf{Loop for each step of episode, } t = T-1, T-2, \dots, 0: \\ \qquad G \leftarrow \gamma G + R_{t+1} \\ \qquad \textbf{Unless } S_t \text{ appears in } S_0, S_1, \dots, S_{t-1}: \\ \qquad \quad \text{Append } G \text{ to } Returns(S_t) \\ \qquad \quad V(S_t) \leftarrow \text{average}(Returns(S_t)) \end{array}
$$
## Blackjack

- Goal to get score higher than dealer without bust.
- Highest possible score is $21$.
- *Bust* is $\gt21$.
- *Natural* is $21$ (ace and 10-card) immediately, player wins.

Scoring:
- 1 through 10 are face value.
- Jack, King, Queen are 10
- Ace are 1 or 11.

Dealer:
- 16 or under: have to take another card.
- 17 or higher: have to stay with their hand.

Player:
- *Hit*: request additional cards
- *Sticks*: stops

State space:
- If we have a *usable* ace, it is always counted.
- If we sum to $\le11$ we always hit.
- Therefore we have:
	- Player sum: \[12-21\]
	- Usable ace: (yes/no)
	- Dealer's card: ace, \[2-10\]

## Summary of Advantages

*Learning from Actual Experience*: It is possible to determine optimal behavior using only sample sequences (experience) without requiring any prior knowledge of the environment's dynamics.
*Learning from Simulated Experience:* The model just needs to generate sample transitions, avoiding the need to calculate complete probability distributions.
*Targeted Estimation:* The computational cost of estimating the value of a single state is independent of the total number of states, allowing you to focus only on a specific subset of states if needed.

## Without a Model

Without a model state-values are insufficient are unable to step one time step forward and choose the best combination of reward and next state. Therefore we must estimate action-values.

The only complications is that many action-pairs may not be visited. For instance, if $\pi$ is a deterministic policy, then only one state-action will be explored by the policy.

Solution one: use *exploring starts*, where we guarantee that episodes start in a state-action pair and that every pair has a nonzero probability of being selected at the start.

Solution two: only consider policies that are *stochastic*, with nonzero probability of selecting all actions in each state.

## Monte Carlo Control (Assuming Exploring Starts)
The process to create approximate optimal policies.

**Assuming infinite episodes and exploring starts**, the Monte Carlo methods will compute $q_{\pi_{k}}$ exactly for arbitrary $\pi_k$.

Given we have action-values, we do not need a model and the policy can be as simple as:
$$
\pi(s) \doteq \text{arg}\max_a q(s,a) \tag{5.1}
$$
Policy improvement is then done by constructing $\pi_{k+1}$ as the greedy policy with respect to $q_{\pi_k}$. 

In practice we may require a large number of episodes to adequately approximate $q_{\pi_k}$, this is mitigated by either:
- Keep track on magnitude and probability of error in estimates, and just keep estimating until they are small enough.
- Give up on completing *policy evaluation* before moving to *policy improvement*. (Like in the previous chapter, where the extreme form of this was value iteration).

For Monte Carlo policy iteration is it natural to perform policy evaluation after each episode, and then perform policy improvement for all states visited in the episode. The pseudocode for such an algorithm, called Monte Carlo Exploring Starts (Monte Carlo ES) is below:
$$
\begin{array}{l} \textbf{Monte Carlo ES (Exploring Starts), for estimating } \pi \approx \pi_* \\ \\ \textbf{Initialize:} \\ \quad \pi(s) \in \mathcal{A}(s) \text{ (arbitrarily), for all } s \in \mathcal{S} \\ \quad Q(s, a) \in \mathbb{R} \text{ (arbitrarily), for all } s \in \mathcal{S}, a \in \mathcal{A}(s) \\ \quad Returns(s, a) \leftarrow \text{empty list, for all } s \in \mathcal{S}, a \in \mathcal{A}(s) \\ \\ \textbf{Loop forever (for each episode):} \\ \quad \text{Choose } S_0 \in \mathcal{S}, A_0 \in \mathcal{A}(S_0) \text{ randomly such that all pairs have probability } > 0 \\ \quad \text{Generate an episode from } S_0, A_0, \text{ following } \pi: S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T \\ \quad G \leftarrow 0 \\ \quad \textbf{Loop for each step of episode, } t = T-1, T-2, \dots, 0: \\ \qquad G \leftarrow \gamma G + R_{t+1} \\ \qquad \text{Unless the pair } S_t, A_t \text{ appears in } S_0, A_0, S_1, A_1 \dots, S_{t-1}, A_{t-1}: \\ \qquad \quad \text{Append } G \text{ to } Returns(S_t, A_t) \\ \qquad \quad Q(S_t, A_t) \leftarrow \text{average}(Returns(S_t, A_t)) \\ \qquad \quad \pi(S_t) \leftarrow \operatorname{argmax}_a Q(S_t, a) \end{array}
$$

## Monte Carlo Control (Without Assuming Exploring Starts)

To avoid the assumption of exploring starts is to somehow guarantee the agent will select all actions. This is done in two ways:

*On-policy* $\rightarrow$ evaluate/improve the policy used to make decisions.
*Off-policy* $\rightarrow$ evaluate/improve a policy different than that used to generate the different.
### On-Policy
The policy starts soft, (i.e. $\pi(a \mid s) > 0$ for all $a$ and $s$) but gradually shifts to be deterministic.
- $\epsilon$-greedy
	- With probability $\epsilon$ choose action at random: $P(a) = \frac{\epsilon}{|\mathcal{A}(s)|}$
	- With probability $1-\epsilon$ choose optimal action: $P(a^*) = 1-\epsilon + \frac{\epsilon}{|\mathcal{A}(s)|}$.
	- Therefore $\pi(a\mid s)>\frac{\epsilon}{|\mathcal{A}(s)|}$ for all $s$ and $a$ for some $\epsilon>0$.

$$\begin{array}{l} \textbf{On-policy first-visit MC control (for } \varepsilon\textbf{-soft policies), estimates } \pi \approx \pi_* \\ \text{Algorithm parameter: small } \varepsilon > 0 \\ \textbf{Initialize:} \\ \quad \pi \leftarrow \text{an arbitrary } \varepsilon\text{-soft policy} \\ \quad Q(s, a) \in \mathbb{R} \text{ (arbitrarily), for all } s \in \mathcal{S}, a \in \mathcal{A}(s) \\ \quad Returns(s, a) \leftarrow \text{empty list, for all } s \in \mathcal{S}, a \in \mathcal{A}(s) \\ \\ \textbf{Repeat forever (for each episode):} \\ \quad \text{Generate an episode following } \pi: S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T \\ \quad G \leftarrow 0 \\ \quad \textbf{Loop for each step of episode, } t = T-1, T-2, \dots, 0: \\ \qquad G \leftarrow \gamma G + R_{t+1} \\ \qquad \text{Unless the pair } S_t, A_t \text{ appears in } S_0, A_0, S_1, A_1 \dots, S_{t-1}, A_{t-1}: \\ \qquad \quad \text{Append } G \text{ to } Returns(S_t, A_t) \\ \qquad \quad Q(S_t, A_t) \leftarrow \text{average}(Returns(S_t, A_t)) \\ \qquad \quad A^* \leftarrow \operatorname{argmax}_a Q(S_t, a) \qquad \qquad \text{(with ties broken arbitrarily)} \\ \qquad \quad \text{For all } a \in \mathcal{A}(S_t): \\ \qquad \qquad \pi(a|S_t) \leftarrow \begin{cases} 1 - \varepsilon + \varepsilon/|\mathcal{A}(S_t)| & \text{if } a = A^* \\ \varepsilon/|\mathcal{A}(S_t)| & \text{if } a \neq A^* \end{cases} \end{array}$$

*Showing how epsilon-greedy policy GPI converges to the optimal epsilon-greedy policy*:

Any $\epsilon$-greedy policy with respect to $q_\pi$ is an improvement over any $\epsilon$-soft policy $\pi$ is shown by the policy improvement theorem. Let $\pi'$ be the $\epsilon$-greedy policy:
$$
\begin{align} q_{\pi}(s, \pi'(s)) &= \sum_{a} \pi'(a|s) q_{\pi}(s, a) \\ &= \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a) + (1 - \varepsilon) \max_{a} q_{\pi}(s, a) \tag{5.2} \\ &\geq \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a) + (1 - \varepsilon) \sum_{a} \frac{\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}(s)|}}{1 - \varepsilon} q_{\pi}(s, a) \\  &= \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a) - \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a) + \sum_{a} \pi(a|s) q_{\pi}(s, a) \\ &= v_{\pi}(s). \end{align}
$$
We can show that the $\epsilon$-greedy policy converges to be optimal by considering an environment which is "optimal" $1-\epsilon$ of the time, then we denote $\tilde{v}_*$ and $\tilde{q}_*$.

Then $\tilde{v}_*$ is equal to:
$$
\begin{align}
\tilde{v}_*(s) &= (1 - \varepsilon) \max_a \tilde{q}_*(s, a) + \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a \tilde{q}_*(s, a) \\
&= (1 - \varepsilon) \max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \tilde{v}_*(s') \right] \\
&\quad + \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \tilde{v}_*(s') \right].
\end{align}
$$
When equality holds and the $\epsilon$-soft policy is no longer improved, then we also know from (5.2):
$$
\begin{align}
v_{\pi}(s) &= (1 - \varepsilon) \max_a q_{\pi}(s, a) + \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a q_{\pi}(s, a) \\
&= (1 - \varepsilon) \max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_{\pi}(s') \right] \\
&\quad + \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_{\pi}(s') \right].
\end{align}
$$
This equation is the same as the original one except for the substitution of $v_\pi$ for $\tilde{v}_*$. Because $\tilde{v}_*$ is the unique solution, it must be that $v_\pi=\tilde{v}_*$.

### Off-Policy
*Important Notation*
- (every visit) $\mathcal{T}(s)$ the set of all time steps in which state $s$ is visited.
- (first visit) $\mathcal{T}(s)$ the set of all time steps in which state $s$ is first visited for all episodes.


Policy being learned about is the *target policy*.
The policy used to generate behavior is the *behavior policy*.

On policy learning can be thought of as a special case of off policy learning where the target and behavior policies are the same.

The assumption of *coverage* assumes that every action for the target policy ($\pi$) is also taken, at least occasionally under the behaviour policy ($b$) ($\pi(a\mid s) > 0$ implies $b(a\mid s)>0$). It follows that $b$ must be stochastic in states where it is not identical to $\pi$.

The target policy is typically the deterministic greedy policy with respect to the current estimate of the action-value function.

So.. How do we compute value functions and action value function for our policy $\pi$ given episodes from another policy $b$?

Given a policy $\pi$, the probability of a specific episode is given by:
$$
\begin{aligned}
\operatorname{Pr}\{A_t, S_{t+1}, A_{t+1}, \dots, S_T \mid S_t, A_{t:T-1} \sim \pi\} 
&= \pi(A_t|S_t)p(S_{t+1}|S_t, A_t)\pi(A_{t+1}|S_{t+1})\cdots p(S_T|S_{T-1}, A_{T-1}) \\
&= \prod_{k=t}^{T-1} \pi(A_k|S_k)p(S_{k+1}|S_k, A_k),
\end{aligned}
$$
 Thus the relative importance of a trajectory under *target* and *behaviour* policies is:
 $$
 \rho_{t:T-1} \doteq \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)p(S_{k+1}|S_k, A_k)}{\prod_{k=t}^{T-1} b(A_k|S_k)p(S_{k+1}|S_k, A_k)} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}. \tag{5.3}
 $$

We can use this to determine the value function for $\pi$ given an episode from $b$:
$$
\mathbb{E}[\rho_{t:T-1} G_t\mid S_t = s] = v_{\pi}(s)
$$

[This is a good video on importance sampling](https://youtu.be/C3p2wI4RAi8?si=q-cvU5tkuueY7QAl).

**Ordinary Importance Sampling**: Determine value function for a policy using a different policy:
$$
V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{|\mathcal{T}(s)|} \tag{5.5}
$$
$$
Q(s,a) \doteq \frac{\sum_{t \in \mathcal{T}(s,a)} \rho_{t+1:T(t)-1} G_t}{|\mathcal{T}(s,a)|}
$$

**Weighted Importance Sampling**: Uses a weighted average instead:
$$
V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}} \tag{5.6}
$$
$$
Q(s,a) \doteq \frac{\sum_{t \in \mathcal{T}(s,a)} \rho_{t+1:T(t)-1} G_t}{\sum_{t \in \mathcal{T}(s,a)} \rho_{t+1:T(t)-1}}
$$
So just to recap. We are still performing GPI with our target policy $\pi$, just when we are going through the evaluation step, we use episodes generated by another policy $b$ and scale the returns such that it fits our target policy $\pi$.

Ordinary importance sampling is unbiased with high variance. Weighted importance sampling is biased with low variance. The weighted importance sampling is generally strongly preferred.




## Summary

Advantages of Monte Carlo methods:
- No transition probabilities required.
- Can be used with complex environments (don't need to compute every possible next state and weight them as in DP methods)
- Easy to focus Monte Carlo methods on a subset of the states.
- Does not bootstrap.
	- TD learning and Dynamic programming bootstrap: $V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1}+\gamma V(S_{t+1}) - V(S_t)]$.
	- This requires the Markov property (prior state is all that is needed to predict future state).
	- Monte Carlo does not bootstrap but waits until the end of the episode and looks at actual return: $V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$

# Chapter 6 Temporal Difference Learning
## TD State Value Prediction
While constant-$\alpha$ Monte Carlo must wait until the end of an episode to update a state value:
$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)] \tag{6.1}$$
Temporal difference learning waits only for the next step:
$$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] \tag{6.2}$$

This is termed TD(0) or *one-step* TD. Written in procedural form:
$$\begin{array}{l} \textbf{Tabular TD(0) for estimating } v_\pi \\ \\ \text{Input: the policy } \pi \text{ to be evaluated} \\ \text{Algorithm parameter: step size } \alpha \in (0, 1] \\ \text{Initialize } V(s), \text{ for all } s \in \mathcal{S}^+, \text{ arbitrarily except that } V(\textit{terminal}) = 0 \\ \\ \text{Loop for each episode:} \\ \quad \text{Initialize } S \\ \quad \text{Loop for each step of episode:} \\ \qquad A \leftarrow \text{action given by } \pi \text{ for } S \\ \qquad \text{Take action } A, \text{ observe } R, S' \\ \qquad V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)] \\ \qquad S \leftarrow S' \\ \quad \text{until } S \text{ is terminal} \end{array}$$
The idea of of basing updates in part on existing estimates is called *bootstrapping* as is done in DP.

To contextualise in terms of equations:
$$
\begin{align}
v_{\pi}(s) &\doteq \mathbb{E}_{\pi}[G_t \mid S_t=s] \tag{6.3} \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t=s] \tag{from 3.9} \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t=s] \tag{6.4} \\
\end{align}
$$
Monte Carlo methods estimate 6.3. They use the full return from the episode.
DP methods estimate 6.4. They only look at the next state.
- DP is an estimate because we do not know $v_\pi(S_{t+1})$ and use $V(S_{t+1})$ instead. We know the environment transition dynamics so $\mathbb{E}_\pi$ is not the issue. (Calculates $\mathbb{E}$ exactly using known probabilities.)
- TD is an estimate because we do not know $v_\pi(S_{t+1})$ and use $V(S_{t+1})$ instead AND the environment transition dynamics are unknown so $\mathbb{E}_\pi$ is also estimated. (Uses one step sample $(R,S′)$ to estimate $\mathbb{E}$.)

6.3 implies an update based on **complete history**.
6.4 implies an update based on **bootstrapping**.

| **Method**          | **Approximates the Expectation (E)?**                                         | **Approximates the Value Function (vπ​→V)?**                                      |
| ------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Monte Carlo         | Yes (Sampling)<br>Uses one sample path to estimate $\mathbb{E}$ in Eq (6.3).  | No (No Bootstrapping)<br>Uses the actual $G_t$, which is unbiased.                |
| Dynamic Programming | No (Full Width)<br>Calculates $\mathbb{E}$ exactly using known probabilities. | Yes (Bootstrapping)<br>Substitutes $V(S_{t+1})$ for $v_\pi(S_{t+1})$ in Eq (6.4). |
| Temporal Difference | Yes (Sampling)<br>Uses one step sample ($R, S'$) to estimate $\mathbb{E}$.    | Yes (Bootstrapping)<br>Substitutes $V(S_{t+1})$ for $v_\pi(S_{t+1})$.             |

TD and Monte Carlo updates are called *sample updates* as they look at a single sample successor state.
DP updates are called *expected updates* as they look at the complete distribution of all possible successors.

In the brackets of the alpha filter above we have what we will term *temporal difference error*:
$$
\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \tag{6.5}
$$
$\delta_t$ is the error in $V(S_t)$ available at time $t+1$.

Monte Carlo error can be written as a sum of TD errors:
$$\begin{align} G_t - V(S_t) &= R_{t+1} + \gamma G_{t+1} - V(S_t) + \gamma V(S_{t+1}) - \gamma V(S_{t+1}) \tag{from (3.9)} \\ &= \delta_t + \gamma(G_{t+1} - V(S_{t+1})) \\ &= \delta_t + \gamma \delta_{t+1} + \gamma^2(G_{t+2} - V(S_{t+2})) \\ &= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} + \dots + \gamma^{T-t-1}\delta_{T-1} + \gamma^{T-t}(G_T - V(S_T)) \\ &= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} + \dots + \gamma^{T-t-1}\delta_{T-1} + \gamma^{T-t}(0 - 0) \\ &= \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k. \tag{6.6} \end{align}$$
If $V$ is updated during the episode, then the identity is not exact. If the step size is small then it may still hold approximately.

## On Policy TD Control
### SARSA
Effectively learn state-action pairs as opposed to state values.
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)] \tag{6.7}
$$
If $S_{t+1}$ is terminal, then $Q(S_{t+1}, A_{t+1})$ is defined as zero.

![[Cheat sheet 4.png|center]]


## Off Policy TD Control
### Q-Learning
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)] \tag{6.8}
$$
![[Cheat sheet 5.png|center]]
### Expected Sarsa
$$
\begin{align}
Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma\mathbb{E}_\pi[Q(S_{t+1}, A_{t+1} \mid S_{t+1}) - Q(S_t, A_t)] \\
&\leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \sum_a \pi(a\mid S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)] \tag{6.9}
\end{align}
$$
- More computational than Sarsa
- Removes variance due to random selection of $A_{t+1}$

Why is this better? Say you are at state A and take action a to reach state B.
In Sarsa, if your next action at B is randomly selected to be a terrible move, the value of your previous state A is massively penalized because of that one unlucky roll.
In Expected Sarsa, the value of state A is updated based on the weighted average of all possible actions at B. It accounts for the risk of the bad move without being skewed by the random occurrence of it.

## Maximisation Bias and Double Learning

Say you have a MDP with states A and B. The reward from B is given by $\mathcal{N}(-0.1, 1)$. Over time Q-learning will explore several actions from B and store their respective rewards. When at A, having transitioned to B Q-learning will take the maximum state-action value from B which also happens to be the randomly better rewards given from B. Therefore the update at A will be biased to the maximum of the Q-values at B.

A possible solution is to store two Q tables. One Q table could be used to pick the maximum action, while the second Q table could have its state-action value for that action as the target for the update of the first Q table. The lookup in Q2 will not have the maximisation bias present in Q1.
This gives the update rule:
$$
Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha [R_{t+1} + \gamma Q_2\left(S_{t+1}, \text{arg}\max_a Q_1(S_{t+1}, a)\right) - Q_1(S_t, A_t)] \tag{6.10}
$$
![[Cheat sheet 6.png|center]]


## Advantages of TD Prediction Models
- No transition probabilities required.
- Can be used with complex environments (don't need to compute every possible next state and weight them as in DP methods).
- Naturally implemented in an online, fully incremental fashion (does not need to wait until end of episode to learn).
- Able to learn from each transition regardless of what subsequent actions are taken (contrast with off policy Monte Carlo methods using importance sampling when the deterministic policy does not choose the given action)
- **TD methods learn faster**.

## Optimality
Monte Carlo methods minimise MSE on existing data.
TD methods minimise  MSE on future data (TD finds estimates that are exactly correct for the maximum-likelihood model of the Markov process).

# N-Step Bootstrapping

## Prediction
*n-step return*:
$$
G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n}) \tag{7.1}
$$
If $t+n \ge T$ (n-step return extends to or beyond termination) then all missing terms are taken as zero and n-step return defined to be equal to the ordinary full return ($G_{t:t+n} \doteq G_t\ \text{if}\ t+n \ge T$).

Therefore the *natural state-value learning algorithm is*:
$$
V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha \left[G_{t:t+n} - V_{t+n-1}(S_t) \right], \qquad 0\le t < T \tag{7.2}
$$
No changes will be made during the first n-1 steps of each episode.

![[Cheat sheet 9.png|center]]

## Control

### N-step Sarsa
Our shift from prediction to control effectively involves switching states for state-action pairs with the addition of an $\epsilon$-greedy policy.

*N-step returns*:
$$
G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^nQ_{t+n-1}(S_{t+n}, A_{t+n}), \qquad n\ge1, 0\le t \lt T-n \tag{7.4}
$$
Therefore the *natural action-state-value learning algorithm is*:
$$
Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha \left[G_{t:t+n} - Q_{t+n-1}(S_t, A_t) \right] \tag{7.5}
$$
![[Cheat sheet 8.png|center]]


N-step Sarsa can be written exactly in terms of novel TD error as:
$$
G_{t:t+n} = Q_{t-1=}(S_t, A_t) + \sum_{k=t}^{\min(t+n, T)-1} \gamma^{k-t}\left[R_{k+1} + \gamma Q_k(S_{k+1}, A_{k+1}) - Q_{k-1}(S_k, A_k) \right] \tag{7.6}
$$

$$
\begin{align*}
G_{t:t+n} &= R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n}) \\
\\
&= R_{t+1} + \gamma Q_t(S_{t+1}, A_{t+1}) - Q_{t-1}(S_t, A_t) + Q_{t-1}(S_t, A_t) \\
&\quad + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n}) - \gamma Q_t(S_{t+1}, A_{t+1}) \\
\\
\text{Let } \delta_t &= R_{t+1} + \gamma Q_t(S_{t+1}, A_{t+1}) - Q_{t-1}(S_t, A_t) \\
\\
&= \delta_t + Q_{t-1}(S_t, A_t) + \gamma \left[ R_{t+2} + \dots + \gamma^{n-2}R_{t+n} + \gamma^{n-1}Q_{t+n-1}(S_{t+n}, A_{t+n}) - Q_t(S_{t+1}, A_{t+1}) \right] \\
\\
&= \delta_t + Q_{t-1}(S_t, A_t) + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} + \dots \\
\\
&= Q_{t-1}(S_t, A_t) + \sum_{k=t}^{\min(t+n, T)-1} \gamma^{k-t} \delta_k
\end{align*}
$$
### Expected Sarsa
Very similar to Sarsa:
$$G_{t:t+n} \doteq R_{t+1} + \dots + \gamma^{n-1}R_{t+n} + \gamma^n \bar{V}_{t+n-1}(S_{t+n}), \qquad t+n < T, \tag{7.7}$$
(with $G_{t:t+n} \doteq G_t$ for $t+n \ge T$) where $\bar{V}_t(s)$ is the _expected approximate value_ of state $s$, using the estimated action values at time $t$, under the target policy:
$$\bar{V}_t(s) \doteq \sum_a \pi(a|s)Q_t(s, a), \qquad \text{for all } s \in \mathcal{S}. \tag{7.8}$$
If $s$ is terminal, then its expected approximate value is defined to be zero.

### n-step Off-policy Learning
When off policy we must once again rescale the return given under the behaviour policy to that of the target policy.

$$V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha \rho_{t:t+n-1} [G_{t:t+n} - V_{t+n-1}(S_t)], \qquad 0 \le t < T, \tag{7.9}$$
where $\rho_{t:t+n-1}$, called the _importance sampling ratio_, is the relative probability under the two policies of taking the $n$ actions from $A_t$ to $A_{t+n-1}$ (cf. Eq. 5.3):
$$\rho_{t:h} \doteq \prod_{k=t}^{\min(h, T-1)} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}. \tag{7.10}$$
As for our n-Step Sarsa update, we have the simple off-policy form:
$$Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha \rho_{t+1:t+n} [G_{t:t+n} - Q_{t+n-1}(S_t, A_t)]  \tag{7.11}$$
![[Cheat sheet 10.png|center]]

## n-Step Tree Backup Algorithm

Effectively taking the expected value over all the next actions like in expected Sarsa but for every step not just the last.

I.e. for the following backup diagram:
- Each first-level action a contributes with a weight of $\pi(a|S_{t+1})$, except that the action actually taken, $A_{t+1}$, does not contribute at all.
- Its probability $\pi(A_{t+1}\mid S_{t+1})$ is used to weight all second-level action values. Thus each non-selected second-level action $a'$ contributes weight $\pi(A_{t+1} \mid S_{t+1})\pi(a'\mid S_{t+2})$.
- And so on.

![[Cheat sheet 11.png|center]]

So in developing the equations, one step tree backup target matches that of Expected Sarsa:
$$
G_{t:t+1} \doteq R_{t+1} + \gamma \sum_a \pi(a\mid S_{t+1})Q_t(S_{t+1},a) \tag{7.15}
$$
for $t \lt T-1$.

And the two step tree-backup return is:
$$
\begin{align*}
G_{t:t+2} &\doteq R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a|S_{t+1})Q_{t+1}(S_{t+1}, a) \\
&\quad\quad + \gamma\pi(A_{t+1}|S_{t+1}) \left( R_{t+2} + \gamma \sum_{a} \pi(a|S_{t+2})Q_{t+1}(S_{t+2}, a) \right) \\
&= R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a|S_{t+1})Q_{t+1}(S_{t+1}, a) + \gamma\pi(A_{t+1}|S_{t+1})G_{t+1:t+2},
\end{align*}
$$
for $t \lt T-2$.

Which we can then turn into a recursive definition:
$$
G_{t:t+n} \doteq R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q_{t+n-1}(S_{t+1}, a) + \gamma \pi(A_{t+1}|S_{t+1}) G_{t+1:t+n}, \tag{7.16}
$$
for $t \lt T-1, n \ge 2$ with $n=1$ being handled by $(7.15)$ except for $G_{T-1:t+n} \doteq R_T$.

This target is then used with the usual action-value update rule from $n$-step Sarsa:
$$Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha \left[ G_{t:t+n} - Q_{t+n-1}(S_t, A_t) \right],$$
for $0 \le t < T$, while the values of all other state–action pairs remain unchanged: $Q_{t+n}(s, a) = Q_{t+n-1}(s, a)$, for all $s, a$ such that $s \neq S_t$ or $a \neq A_t$.

![[Cheat sheet 12.png|center|half]]


## Generalising

We can generalise n-step Sarsa, expected Sarsa and tree-backup Sarsa into one algorithm: n-step $Q(\sigma)$.

![[Cheat sheet 13.png|center|half]]
>[!error] TODO
> Go over equations used

The algorithm:
![[Cheat sheet 14.png|center|half]]

# Planning
## Models and Planning
Types of models:
- Distribution models: produce a description of all possibilities and their probabilities (used in dynamic programming).
- Sample models: produce just one of the possibilities sampled according to the probabilities.

Online planning: performed while interacting with the environment.

Within a planning agent, there are at least two roles for real experience:
- it can be used to improve the model (*model learning*)
- it can be used to improve the value function and policy (*direct reinforcement learning*)

Therefore experience can improve value functions and policy either directly or indirectly through the model (called *indirect reinforcement learning*).

Q-learning vs random sample one-step tabular Q-planning. In the latter we draw a random state and action, perform one update, then repeat - we don't finish the episode.

## Dyna-Q
Dyna-Q:
- Planning: random-sample one-step tabular Q-planning. Only samples from state-action pairs that have been previously experienced.
- Model-learning: after each experienced transition, the model records the transition in a table entry for $S_t, A_t$ and the experienced $R_{t+1}, S_{t+1}$. Therefore if the model is queried with a state-action pair that has been experienced before, it simply returns the last-observed next state and next reward as its prediction.
- *search control* the act of selecting starting state and actions for simulated experiences generate by the model.
- Direct-RL: one-step tabular Q-learning

In other words - Dyna-Q:
1. Planning - random-sample one-step tabular Q-planning
	- Select state and action at random, send them to a sample model, generate return
	- *search control* the act of selecting starting state and actions for simulated experiences generate by the model.
2. Direct-RL - one-step tabular Q-learning
	- After a real transition in the environment, observe result $S_t, A_t \rightarrow R_{t+1}, S_{t+1}$, update policy/value function.
3. Model Learning - tabular - $M(S_t, A_t) \leftarrow R_{t+1}, S_{t+1}$
	- Used for planning

![[Cheat sheet 15.png|center]]
In pseudocode:
![[Cheat sheet 16.png|center]]

Dyna-Q+:
- If the modeled reward for a transition is $r$ and the transition has not been tried in $\tau$ time steps, then **planning** updates are done as if that transition produced a reward of $r+k\sqrt{\tau}$.
- The idea is to encourage exploration during **direct-RL**.


During planning, we may end up sampling state-action pairs that are not very useful. For instance in a maze world, after the first episode direct-RL will have only stored one state-action pair value. For that value to be propagated, state-action that immediately lead to that state will need to be selected - this may not be a common occurrence. We need a way to prioritise updates of state-actions immediately preceding state-actions that have been recently updated - we need a measure of urgency. This idea is called *prioritised sweeping*.

For e.g., priority: $P=|r+\gamma \max_a Q(S_{t+1},a) - Q(S_t, A_t)|$.

![[Cheat sheet 17.png|center]]

## Expected vs Sample Updates
Recall sample vs expected updates for $q_*$:
- Expected update:
$$Q(s,a) \leftarrow \sum_{s',r} \hat{p}(s',r \mid s,a)[r+\gamma \max_{a'} Q(s', a')] \tag{8.1}$$
- Sample update (Q-learning-like update)
$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[R+\gamma \max_{a'}Q(S', a') - Q(s,a) \right] \tag{8.2}$$
Expected updates are an exact computation - correctness is limited only by the correctness of $Q(s',a')$ at the successor states while sample updates also suffer from sampling error.

The sample update is however cheaper computationally. Let $b$ be the *branching factor* (number of possible next states $s'$ for which $\hat{p}(s' \mid s,a)\gt 0$), then the expected update of a state-action pair requires around $b$ times as much computation.

For very high branching factors, sample updates appear far more preferable: (sample updates reduce error according to $\sqrt{\frac{b-1}{bt}}$ where $t$ is the number of sample updates that have been performed).
![[Cheat sheet 18.png|center]]


## Trajectory Sampling
>[!error] TODO
## Heuristic Search

In the context of Reinforcement Learning (RL), Heuristic Search is essentially Decision-Time Planning.

Instead of just "generating episodes" (which implies running blindly until the end), heuristic search usually builds a tree of possibilities starting from your current state St​.
- You look ahead: "If I do A, then B happens. If I do C, then D happens."
- You explore the immediate future deeply, rather than the distant past.

"Updating our Q table" -> Backing Up Values
- In standard Q-learning, we update values based on one step of real experience.
- In Heuristic Search, we go deep into the tree, find the values at the leaf nodes (often using our existing, approximate value function), and then back them up to the root.
- This gives the current state St​ a highly accurate value estimate, much better than the "average" estimate stored in your Q-table.



# Function Approximation
We now move from a tabular value function to a parameterised functional form with weight vector $\mathbf{w} \in \mathbb{R}^d$. Such that $\hat{v}(s, \mathbf{w}) \approx v_\pi(s)$.  

## Value-function Approximation
Let us redefine the notation of $v(S_{t}) \leftarrow v(S_{t}) + \alpha[ R_{t+1} + \gamma V(S_{t+1}) -V(S_t)]$ as $v(S_t) \mapsto R_{t+1} +\gamma V(S_{t+1})$.

## The Prediction Objective ($\overline{VE}$)
Let us define a state distribution $\mu(s)\ge 0, \sum_s \mu(s)=1$ representing how much we care about the error in each state s.

Weighting over the state space by $\mu$ we obtain a natural objective function the *Mean Squared Value Error* denoted $\overline{VE}$:
$$
\overline{VE}\doteq \sum_{s\in\mathcal{S}} \mu(s) \left[v_\pi(s) - \hat{v}(s, \mathbf{w}) \right]^2 \tag{9.1}
$$
Often $\mu(s)$ is chosen to be the fraction of time spent in $s$. (Called the *on-policy distribution*). For continuing tasks $\mu(s)$ is chosen to be a stationary distribution. If we had a nonstationary distribution in continuing tasks the target would constantly vary making convergence very difficult and potentially leading to catastrophic forgetting.

Let:
- $h(s)$ be the probability that an episode begins in each state $s$.
- $\eta(s)$ be the number of time steps spent on average in a state $s$ in a single episode.
Then time is spent in a state $s$ if the episode starts in $s$ or if transitions are made into $s$ from a preceding state $\bar{s}$:
$$
\eta(s) = h(s) + \sum_{\bar{s}} \eta(\bar s)\sum_a \pi(a \mid \bar s)p(s \mid \bar s, a),\qquad \text{for all}\ s \in \mathcal S \tag{9.2}
$$
The on-policy distribution is the fraction of time spent in each state normalised to sum to one:
$$
\mu(s) = \frac{\eta(s)}{\sum_{s'} \eta(s')} \tag{9.3}
$$
In the context of discounting, we have:
$$
\eta(s) = h(s) + \gamma \sum_{\bar{s}} \eta(\bar s)\sum_a \pi(a \mid \bar s)p(s \mid \bar s, a),\qquad \text{for all}\ s \in \mathcal S
$$

## Stochastic-gradient and Semi-gradient Methods

We adjust the weight vector after each example by a small amount in the direction that would most reduce the error on that example:
$$
\begin{align}
\mathbf{w}_{t+1} &\doteq \mathbf{w}_t - \frac{1}{2} \alpha \nabla\left[ v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t) \right]^2 \tag{9.4}\\
&= \mathbf{w}_t + \alpha \left[v_\pi (S_t) - \hat{v}(S_t, \mathbf{w}_t) \right] \nabla \hat{v}(S_t, \mathbf{w}_t) \tag{9.5}
\end{align}
$$
So you may note that for our purposes we don't have $v_\pi(S_t)$ and therefore will be relying on bootstrapping ($R_{t+1} +\gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$), but in the 9.5 we don't actually take the derivative of the target. The reason we do this is for simplicity. Because we're not fully differentiating the target we call this a *semi-gradient* method. More on this:

We now turn to the case where the target output $U_t \in \mathbb{R}$ of the $t$th training example is not the true value $v_\pi(S_t)$ but some approximation to it. For e.g., $U_t$ may be a noise-corrupted version of v$_\pi(S_t)$ or some bootstrapping target. In these cases we cannot perform the exact update (9.5) as $v_\pi(S_t)$ is unknown. This yields the following general SGD method for state-value prediction:
$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \left[U_t - \hat{v}(S_t, \mathbf{w}_t)\right] \nabla \hat{v}(S_t, \mathbf{w}_t) \tag{9.7}
$$
If $U_t$ is an _unbiased_ estimate, that is, if $\mathbb{E}[U_t | S_t = s] = v_{\pi}(S_t)$, for each $t$, then $\mathbf{w}_t$ is **guaranteed** to converge to a local optimum under the usual stochastic approximation conditions (2.7) for decreasing $\alpha$.

The Monte-Carlo target $U_t \doteq G_t$ is by definition an unbiased estimate of $v_\pi(S_t)$. Pseudocode below:
![[Cheat sheet 19.png|center|half]]

The same **guarantees** are not obtained if a bootstrapping estimate of $v_\pi(S_t)$ is used. Bootstrapping or DP methods all depend on the current value of the weight vector $\mathbf{w}_t$ which implies that they will be biased and will not produce a true gradient-descent method. Accordingly they are called *semi-gradient* methods.

While *semi-gradient* (bootstrapping) methods do not converge as robustly as gradient methods, they do converge reliably in important cases (such as the linear case) as well as converging much quicker (as seen in TD learning). They also allow learning to be continual and online, without waiting for the end of an episode.

A prototypical *semi-gradient* method is TD(0): $U_t \doteq R_{t+1} + \gamma \hat v (S_{t+1}, \mathbf{w})$:
![[Cheat sheet 20.png|center|half]]
## Linear Methods
We define a function $\mathbf{x}$: $x_i : \mathcal S \rightarrow \mathbb{R}$. I.e., $x_i(s)$ extracts feature $i$ from the state $s$.

We then define the state-value function under linear methods as the inner product between $\mathbf w$ and $\mathbf x(s)$:
$$
\hat v(s, \mathbf w) \doteq \mathbf w^\top \mathbf x(s) \doteq \sum_{i=1}^d w_i x_i(s) \tag{9.8}
$$
Then the gradient of the approximate value function with respect to $\mathbf w$ is:
$$
\nabla \hat{v}(s, \mathbf w) = \mathbf x(s)
$$
Thus in the linear case the general SGD update (9.7) reduces to:
$$
\mathbf w_{t+1} \doteq \mathbf w_t + \alpha \left[U_t - \hat v(S_t, \mathbf w_t) \right] \mathbf x(S_t)
$$
Moreover in the linear case there is only one optimum, therefore any method that is guaranteed to converge to or near a local optimum is automatically guaranteed to converge to or near the global optimum.

Expanding out 9.8:
$$\begin{align} 
\mathbf{w}_{t+1} &\doteq \mathbf{w}_t + \alpha \left( R_{t+1} + \gamma \mathbf{w}_t^\top \mathbf{x}_{t+1} - \mathbf{w}_t^\top \mathbf{x}_t \right) \mathbf{x}_t \tag{9.9} \\ 
&= \mathbf{w}_t + \alpha \left( R_{t+1} \mathbf{x}_t - \mathbf{x}_t ( \mathbf{x}_t - \gamma \mathbf{x}_{t+1} )^\top \mathbf{w}_t \right), 
\end{align}$$

When we reach steady state, the expected next weight vector can be written as:
$$\mathbb{E}[\mathbf{w}_{t+1} | \mathbf{w}_t] = \mathbf{w}_t + \alpha(\mathbf{b} - \mathbf{A}\mathbf{w}_t) \tag{9.10}$$
where
$$\mathbf{b} \doteq \mathbb{E}[R_{t+1} \mathbf{x}_t] \in \mathbb{R}^d \quad \text{and} \quad \mathbf{A} \doteq \mathbb{E}[\mathbf{x}_t (\mathbf{x}_t - \gamma \mathbf{x}_{t+1})^\top] \in \mathbb{R}^d \times \mathbb{R}^d \tag{9.11}$$
From (9.10) it is clear that, if the system converges, it must converge to the weight vector $\mathbf{w}_{\text{TD}}$ at which
$$\begin{aligned} 
\mathbf{b} - \mathbf{A}\mathbf{w}_{\text{TD}} &= \mathbf{0} \\ 
\Rightarrow \mathbf{b} &= \mathbf{A}\mathbf{w}_{\text{TD}} \\ 
\Rightarrow \mathbf{w}_{\text{TD}} &\doteq \mathbf{A}^{-1} \mathbf{b}. 
\end{aligned} \tag{9.12}$$
**That is to say** that TD converges to a fixed point (the *TD fixed point*).

> [!error]
> Might be worth going over the Proof of Convergence of Linear TD(0)























# Summary

## Prediction
### Monte Carlo
**On Policy Monte Carlo**
$$
\begin{align}
N &\leftarrow N + 1 \\
V(S_t) &\leftarrow V(S_t) + \frac{1}{N} [G_t - V(S_t)] \tag{6.1}
\end{align}
$$
**Constant-$\alpha$ Monte Carlo**
$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)] \tag{6.1}$$
**Off Policy Monte Carlo** 
Note $W$ updated first as the return is dependent on the next action.
$$
\begin{align}
W &\leftarrow W\cdot\frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)} \\
C(S_t) &\leftarrow C(S_t) + W \\
V(S_t) &\leftarrow V(S_t) + \frac{W}{C(S_t)} [G_t - V(S_t)] \tag{6.1}
\end{align}
$$
### TD
**Temporal Difference Learning**
$$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] \tag{6.2}$$
## Control
### Monte Carlo
**On Policy Monte Carlo**
$$
\begin{align}
N(S_t, A_t) &\leftarrow N(S_t, A_t) + 1 \\
Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} [G_t - Q(S_t, A_t)]
\end{align}
$$
**Constant-$\alpha$ Monte Carlo**
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [G_t - Q(S_t, A_t)]
$$

**Off Policy Monte Carlo**
Note $W$ is updated **after** the $Q$ update here. This is because $Q(S_t​,A_t​)$ estimates the value conditional on taking $A_t​$, so we do not re-weight the probability of At​ occurring, we only re-weight the subsequent trajectory.
$$
\begin{align}
C(S_t, A_t) &\leftarrow C(S_t, A_t) + W \\
Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \frac{W}{C(S_t, A_t)} [G_t - Q(S_t, A_t)] \\
W &\leftarrow W\cdot\frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}
\end{align}
$$
### TD
#### On Policy
**SARSA**
$$
Q(S_t,A_t) \leftarrow Q(S_t, A_t) + \alpha\left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]
$$
**Q-Learning**
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma\max_{a\in\mathcal{A}}Q(S_{t+1}, a) - Q(S_t, A_t) \right]
$$
**Expected SARSA**
$$
Q(S_t,A_t) \leftarrow Q(S_t, A_t) + \alpha\left[ R_{t+1} + \gamma \mathbb{E}_\pi [Q(S_{t+1}, A_{t+1})] - Q(S_t, A_t) \right]
$$


















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





