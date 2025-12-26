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
$$G_t \doteq R_{t+1}, R_{t+2}, R_{t+3}, \dots, R_T \tag{3.7}$$
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
\mathbb{E}[\rho_{t:T-1} \mid S_t = s] = v_{\pi}(s)
$$

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





