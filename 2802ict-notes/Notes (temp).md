
### Broad Implementation Notes
f() represents the cost of a node.

### General Implementation

Note how as soon as we discover a new node, we record it in reached and push it onto the frontier.

Best First Search:
```python
def best_first_search(problem, f) -> solution or failure:
    node = Node (state = problem.initial)
    frontier = PriorityQueue() # ordered by f
    reached = Map() # lookup table by state

    while frontier not empty:
        node = frontier.pop()
        if problem.is_goal(node) return node

        for child in expand(problem, node):
            s = child.state
            if s not in reached or s.cost < reached[s].cost:
                reached[s] = child
                frontier.add(child)
    return failure

def expand(problem, node) -> Iterable[node]:
    s = node.state
    for action in problem.actions(s):
        s_prime = problem.result(s)
        cost = node.path_cost + problem.action_cost(s, action, s_prime)
        yield Node(state=s_prime, parent=node, action=action, path_cost=cost)
```

A* Search (admissable):
```python
def best_first_search(problem, f) -> solution or failure:
    node = Node (state = problem.initial)
    frontier = PriorityQueue() # ordered by f (**f contains heuristic**)
    reached = Map() # lookup table by state

    while frontier not empty:
        node = frontier.pop()
        if problem.is_goal(node) return node

        for child in expand(problem, node):
            s = child.state
            if s not in reached or s.cost < reached[s].cost:
                reached[s] = child
                frontier.add(child)
    return failure

def expand(problem, node) -> Iterable[node]:
    s = node.state
    for action in problem.actions(s):
        s_prime = problem.result(s)
        cost = node.path_cost + problem.action_cost(s, action, s_prime)
        yield Node(state=s_prime, parent=node, action=action, path_cost=cost)
```

A* Search (consistent):
```python
def best_first_search(problem, f) -> solution or failure:
    node = Node (state = problem.initial)
    frontier = PriorityQueue() # ordered by f (**f contains heuristic**)
    reached = Map() # lookup table by state
    expanded = Set()

    while frontier not empty:
        node = frontier.pop()
        if problem.is_goal(node) return node
        expanded.add(node)

        for child in expand(problem, node):
            s = child.state
            if s not in expanded and (s not in reached or s.path_cost < reached[s].cost):
                reached[s] = child
                frontier.add(child)
    return failure

def expand(problem, node) -> Iterable[node]:
    s = node.state
    for action in problem.actions(s):
        s_prime = problem.result(s)
        cost = node.path_cost + problem.action_cost(s, action, s_prime)
        yield Node(state=s_prime, parent=node, action=action, path_cost=cost)
```