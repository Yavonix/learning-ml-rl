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
The *greedy choice is baked directly into the update rule*:


No need for discrete policy improvement and evaluation steps.












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