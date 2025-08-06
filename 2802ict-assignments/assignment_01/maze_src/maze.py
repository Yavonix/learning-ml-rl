import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable

from common import (
    DepthLimitReachedError,
    NoSolutionError,
    NoSolutionError,
    Node,
    State,
    Maze_Common,
)


@dataclass
class StackFrontier:
    frontier: list[Node] = field(default_factory=list)

    def __init__(self, initial_nodes: list[Node] | None = None):
        if initial_nodes:
            self.frontier = initial_nodes

    def __contains__(self, state: State) -> bool:
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def add(self, node: Node):
        self.frontier.append(node)

    def extend(self, node: list[Node] | Iterable[Node]):
        self.frontier.extend(node)

    def remove(self, node: Node):
        """Remove node from frontier"""
        self.frontier.remove(node)

    def pop(self) -> Node:
        """Pop next node from frontier"""
        if self.empty():
            raise Exception("empty frontier")
        else:
            return self.frontier.pop()

    def size(self) -> int:
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


if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze.txt")

if __name__ == "__main__":

    m = Maze(sys.argv[1])

    m.reset_stats()
    try:
        m.ids_solve()
    except NoSolutionError:
        pass
    except Exception as e:
        raise e
    m.print()

    # print("Solution:\n\n")
    # m.output_image("maze.png", show_explored=True)
