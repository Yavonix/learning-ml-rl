import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable
import heapq

from common import (
    DepthLimitReachedError,
    NoSolutionError,
    NoSolutionError,
    Node,
    State,
    Maze_Common,
    FrontierADT
)

class StackFrontier(FrontierADT):
    def __init__(self, initial_nodes: list[Node] | None = None):
        if initial_nodes:
            self.frontier = initial_nodes
        else:
            self.frontier = []

    def __contains__(self, state) -> bool:
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def add(self, node):
        self.frontier.append(node)

    def extend(self, nodes):
        self.frontier.extend(nodes)

    def remove(self, node):
        """Remove node from frontier"""
        self.frontier.remove(node)

    def pop(self):
        """Pop next node from frontier"""
        if self.empty():
            raise Exception("empty frontier")
        else:
            return self.frontier.pop()

    def size(self):
        return len(self.frontier)

class PriorityQueueFrontier(FrontierADT):
    def __init__(self, cost_fn: Callable[[Node], int], initial_nodes: list[Node] | None = None):
        self.frontier: list[tuple[int, Node]] = []
        self.score = cost_fn
        if initial_nodes:
            for node in initial_nodes:
                heapq.heappush(self.frontier, (self.score(node), node))
    
    def __contains__(self, state) -> bool:
        return any(node.state == state for (priority, node) in self.frontier)
    
    def empty(self):
        return len(self.frontier) == 0
    
    def add(self, node):
        heapq.heappush(self.frontier, (self.score(node), node))

    def extend(self, nodes):
        for node in nodes:
            heapq.heappush(self.frontier, (self.score(node), node))

    def remove(self, node):
        self.frontier.remove((self.score(node), node))
        heapq.heapify(self.frontier)

    def pop(self):
        return heapq.heappop(self.frontier)[1]
    
    def size(self):
        return len(self.frontier)


class Maze(Maze_Common):

    def dfs_solve(self, max_depth=float("inf")) -> None:
        """Depth-limited depth-first search"""

        start_node = Node(state=self.start, parent=None, action=None, depth=0)
        frontier = StackFrontier(initial_nodes=[start_node])

        self.explored = set()

        cutoff_occurred = False

        while not frontier.empty():
            node = frontier.pop()
            self.num_explored += 1

            if node.state == self.goal:
                self.solution = node
                return

            self.explored.add(node.state)

            for neighbour in self.neighbors(node):
                if neighbour.depth > max_depth:
                    cutoff_occurred = True
                    continue
                if neighbour.state not in self.explored and neighbour.state not in frontier:
                    frontier.add(neighbour)

        if cutoff_occurred:
            raise DepthLimitReachedError
        else:
            raise NoSolutionError

    def ids_solve(self) -> None:
        """Depth-limited iterative depth-first search"""

        depth = 1
        max_depth = 100

        while depth <= max_depth:
            try:
                self.dfs_solve(max_depth=depth)
                return
            except DepthLimitReachedError:
                depth += 1
            except Exception as e:
                raise e

        raise DepthLimitReachedError
    
    def manhattan_heuristic(self, node: Node) -> int:
        return abs(self.goal[0]-node.state[0]) + abs(self.goal[1]-node.state[1])
    
    def astar_solve(self) -> None:
        start_node = Node(state=self.start, parent=None, action=None, depth=0)
        frontier = PriorityQueueFrontier(cost_fn=lambda n: n.depth + self.manhattan_heuristic(n), initial_nodes=[start_node]) # decides order
        reached: dict[State, Node] = {start_node.state: start_node} # have I seen this state before, and was it via a better path?
        explored: set[State] = set() # prevents reexpansion for consistent heuristic

        while not frontier.empty():
            node = frontier.pop()
            self.num_explored += 1

            if node.state == self.goal: # assuming heuristic is consistent
                self.solution = node
                return
            
            explored.add(node.state)
            
            for neighbour in self.neighbors(node):
                s = neighbour.state
                if s not in explored and (s not in reached or neighbour.depth < reached[s].depth):
                    reached[s] = neighbour
                    frontier.add(neighbour)


if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze.txt")

if __name__ == "__main__":

    m = Maze(sys.argv[1])

    m.reset_stats()
    try:
        m.astar_solve()
    except NoSolutionError:
        pass
    except Exception as e:
        raise e
    m.print()

    # print("Solution:\n\n")
    # m.output_image("maze.png", show_explored=True)
