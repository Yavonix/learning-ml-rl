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
- $\mathcal{S}^+$ all states excluding the **terminal state**
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
&= \max_{a} \mathbb{E}_{\pi_*} [ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a ] & \tag{by (3.9)} \\
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
Refreshing the Bellman optimality equations:
$$
\begin{align} v_*(s) &= \max_{a} \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t=s, A_t=a] \\ &= \max_{a} \sum_{s',r} p(s',r|s,a)[r + \gamma v_*(s')], \text{ or} \tag{4.1} \\
q_*(s,a) &= \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t=s, A_t=a\right] \\ &= \sum_{s',r} p(s',r|s,a)\left[r + \gamma \max_{a'} q_*(s', a')\right], \tag{4.2}
\end{align}
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