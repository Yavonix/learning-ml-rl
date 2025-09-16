## Week 2 Notes

### Agent Types
Types of agents:
- Simple reflex agents
- Model-based reflex agents
- Goal-based agents
- Utility-based agents

### Algorithm overview

- Completeness: Does the algorithm find a solution where one exists?
- Cost optimality: Does the algorithm find the solution with the lowest path cost?
- Time complexity…
- Space complexity…
  - State space graph (explicit) measured by $|V|+|E|$ where V is graph vertices and E is edges. 
  - State space graph (implicit) measured by
    - **d**: shallowest solution depth (optimal solution)
    - **m**: max solution depth
    - **b**: branch factor/the number of successors per node

**Best first search** is an umbrella term for any search that uses a priority queue ordered by an evaluation function f(n).

Let g(n) be path cost from initial to node.
Let h(n) be heuristic cost.

Uninformed algorithms: Has no heuristic function. 
- BFS: uses fifo queue.
  - Optimal if all step cost = 1
  - Can use early goal test.
- DFS: uses lifo stack.
  - Not optimal
  - Can use early goal test.
- UCS: uses priority queue. f(n) = g(n)
  - Optimal when f(n) $\ge 0$.
  - Also called Dijkstra's algorithm
- IDS: uses lifo stack
  - Iterative deepening search
  - In general, is the preferred uninformed search memory when search state space is larger than can fit into memory and solution depth is not known.
  - Expand the shallower frontier all the time


DLS:              ancestor check only
Tree DFS:         ancestor check only  
Graph DFS:        explored set
Tree BFS:         ancestor check only (never used)
Graph BFS:        frontier + explored sets
A* consistent:    frontier + explored sets
A* inconsistent:  reached states dict with cost comparison

Informed algorithms: Has a heuristic function.
- Greedy search: uses priority queue. f(n) = h(n)
- A*: uses priority queue. f(n) = g(n) + h(n)
  - Cannot use early goal test regardless of consistent/admissible.
- Weighted A*: f(n) = g(n) + $\epsilon$h(n)

Open loop: agent does not access percepts during execution
Closed loop: agent does access percepts during execution

A node: corresponds to a state in the state space
- Parent
- Action
- State
When we expand a node we determine the next possible states. Nodes returned from exploring represent the frontier of the search tree.

## Week 3 Stuff

### Anatomy of A*

A* is a type of best-first search that utilises path cost g(n) and heuristic cost h(n).

Admissible heuristic: a heuristic where estimated cost is always $\le$ to true cost.
Consistent heuristic: a heuristic where estimated cost is always $\le$ to (step cost to another node n_prime + the heuristic of n_prime)

Consistency implies admissibility.


#### Admissible A*

Required datastructures:
- Priority queue
- Reached map (Node -> Cost)
Neighbour check: not in reached or node.f_cost < reached[node].f_cost

Admissibility only guarantees any explored nodes n are cost optimal when its known path cost g(n) is no greater than the best possible completion cost still lurking on the frontier. That is,

$$
g(n)\;\le\;\min_{m\in \text{OPEN}}\bigl(g(m)+h(m)\bigr).
$$

This means that the **exit condition** for admissible A* must check if the node matches the goal state **and** the cost of the node is less than the cost of any node in the frontier. A side effect of this is that the algorithm must keep track of the best candidate solution so far.

Extra:
- Cost optimality guaranteed even if only the nodes on solution path are admissible.
- Cost optimality guaranteed even if heuristic is not admissibility but never overestimates a cost more than the difference between the optimal and second most optimal solutions.

#### Consistent A*

Required datastructures:
- Priority queue
- Reached map (Node -> Cost)
- Explored (prevent reexpansions)
Neighbour check: not in explored and (not in reached or reached[node].f_cost < node.f_cost)

Consistency guarantees that any explored nodes n are cost optimal.

That means our **exit condition** must only check if the node matches the goal state.

In addition, it means we never need to re-expand a pre-expanded node. Therefore we can introduce an `expanded` set variable and before appending to the frontier, we can check if the node has already been expanded or not.

#### Weighted A*

$$
f(n)=g(n)+\epsilon\,h(n)
$$
for some weight $\epsilon>1$.

This will:
- Decrease runtime as search is more heavily weighted towards nodes "closer" to the goal.
- Lose admissibility and hence cost-optimality.

However, weighted A* does provide a suboptimality guarantee that the cost C of the returned solution satisfies $C \;\le\; \epsilon \,C^*$

In weighted A* we can also get rid of the $g(n)\;\le\;\min_{m\in \text{OPEN}}\bigl(g(m)+h(m)\bigr)$ check because we've given up on cost-optimality. Getting rid of the check will reduce solution quality (suboptimality still guaranteed).

#### Beam Search

An alternative form of A* where cost optimality and suboptimality are sacrificed for memory and performance. The frontier is limited to either:
- Only keeping the lowest cost k nodes.
- Only keeping nodes with costs within $\delta$ of current optimal node. 

## Week 4 Stuff

### Anatomy of Local Search

**Local search** refers to a family of optimisation methods that explore a problem’s solution space by iteratively moving to "neighbour" states

Given:
- A set of states
- An evaluation function
Find $X^*$ where $\forall X, Eval(X^*)\gt Eval(X)$

We do not care about actions taken.

#### Hill Climbing Search

**Hill climbing** refers to the general framework of moving to a better neighbour until no improvement is possible.

Incomplete as it can get stuck in local maximums.

There are several variants:

- **Greedy Local Search / Steepest Ascent Hill Climb**

Choose highest value neighbour until reached local/global maximum.

```python
def hill_climb(problem) -> state:
  current = make_node(problem.initial_state)
  while True:
    neighbour = highest_value_neighbour(current)
    if eval(neighbour) > eval(current): current = neighbour
    else: return current
```

We may allow sideway moves to escape from shoulders (and limit max number of consecutive sideway moves allowed).

- **Stochastic Hill Climb**

Choose neighbour at random from uphill moves.

- **First-choice Hill Climb**

Choose first randomly generated neighbour which is uphill. Good if node has thousands of neighbours.

- **Random-restart Hill Climb**

Perform hill-climb multiples times with random starting locations.

If $p$ is probability of success on first try. The expested number of restarts is $1/p$.

#### Simulated Annealing

For K times:
  Start with high temperature T, iterate over neighbours randomly and:
  - if eval(neighbour) > eval(current), accept the move
  - else accept move with probability $e^{-(E-E')/T}$
  T = $\alpha \cdot T$ ($\alpha<1$)

Large differences in eval cost (E-E') reduce the likelihood of picking the downhill move.
T decreases with time to also reduce the likelihood of picking the downhill move.

### Constraint Satisfaction Problems

**State** defined by variables $X_i$ with values from domain $D_i$.
**Goal test** a set of constraints specifying allowable combinations of values for variables.

**Unary constraint**: restricts the domain of a single variable.
**Binary constraint**: restricts the allowable value combinations of a pair of variables.
**Higher order constraints**: 3 or more variables.

For $n$ variables, $d$ domain size and $l$ depth, branching factor is $b=(n-l)\cdot d$

Over n levels (ℓ=0…n–1) the total number of leaves is

$$
\prod_{\ell=0}^{n-1}\bigl((n-\ell)d\bigr) \;=\; (n\cdot(n-1)\cdots1)\,\times\,d^n \;=\; n!\,\cdot\,d^n.
$$

#### Backtracking Search

Basic uninformed algorithm for solving CSPs.

```python
def backtrack(assignments, constraints):
  if complete(assignments): return assignments
  var = select_unassigned_variable(assignments, constraints)
  for value in domain_values(var, assignments, constraints):
    if consistent(value, assignments):
      assignments.add(var, value)
      result = backtrack(assignments, constraints)
      if result != failure: return result
      assignments.remove(var, value)
  return failure
```

We can add heuristics to optimise search:
- Minimum remaining values: pick the node with the smallest domain
- Degree heuristic: pick the node with the most constraints
- Least constrained value: assign the value with the least constraints
- Forward chaining: after each assignment, immediately eliminate inconsistent values from neighbouring domains

#### Arc Consistency

A node is arc consistent iff for every value x of X, there is some allowed y.

```python
def ac3(constraints) -> constraints:
  queue = []
  while queue:
    x_i, x_j = remove_first(queue)
    if remove_inconsistent_values(x_i, x_j):
      for k in neighbours(x_i):
        queue.append(k, x_i)

# very very very rough pseudocode
def remove_inconsistent_values(x_i, x_j) -> bool:
  removed = false
  for x in domain(x_i):
    support_exists = False
    for y in domain(x_j):
      if constraint(x_i, x, x_j, y):
        support_exists = True
        break
    if not support_exists:
      domains[Xi].remove(x)
      removed = True
  return removed
```

## Week 5 Stuff

### Classification of Machine Learning

Types of machine learning:
- Supervised (examples and labels given)
  - Classification (discrete predictions)
  - Regression (continuous predictions)
- Unsupervised (only examples, no labels)
  - Clustering
  - Non-clustering
    - PCA: Linear transform that finds orthogonal directions (principal components) that maximise variance (extract the directions of greatest variance, which is very useful for feature extraction, denoising and compression)
    - ICA: Linear transform that finds components that are statistically independent. (separate mixed sources (blind source separation))
  - Association (rule) learning (wikipedia classifies this as supervised learning)
    - $X\implies Y$
    - Every rule is composed by two different sets of items, also known as itemsets, X and Y, where X is called antecedent or left-hand-side (LHS) and Y consequent or right-hand-side (RHS)
- Reinforcement learning (learn sequence of actions to maximise payoff)
  - Algorithm/agent interacts with the environment and gets a positive/negative reward
  - Common RL algorithms:
    - Temporal difference
    - Deep adversarial networks
    - Q-learning
  - Easier to work with when dealing with unlabeled data sets
  - Most ML platforms don't have reinforcement learning capabilities because they require higher computing power than most organisations have.

Trade-off between three factors (Dietterich, 2003)
- Model class complexity (hypothesis-class complexity)
- Training data size
- Generalisation error

- As training data size increases -> generalisation error decreases
- As complexity increases -> generalisation error decreases then increases

To estimate generalization error, we need data unseen during
training. We split the data as
  - Training set (60-80%)
  - Validation set (10-20%)
  - Test (publication) set (10-20%)