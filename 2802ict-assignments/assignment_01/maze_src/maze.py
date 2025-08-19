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

import numpy as np
from cnn_heuristic.dataset import (
    generate_object_map, 
    generate_euclid_transform_map,
    gen_random_goal,
    generate_goal_map,
    convert_bool_to_obstacle_map,
    LABEL_NORMALISER
)
from cnn_heuristic.model import Encoder_Decoder
from flax import nnx
import orbax.checkpoint as ocp


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
        """Iterative depth-first search"""

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
    
    def cnn_heuristic(self, node: Node) -> int:
        assert self.width == self.height == 224

        if self.heuristic_cache == None:
            print("Generating heuristic map...")
            rngs = nnx.Rngs(params=0)
            checkpointer = ocp.StandardCheckpointer()
            checkpoint_dir = "/home/roman/learning-ml/2802ict-assignments/assignment_01/maze_src/cnn_heuristic/checkpoints/save-no-normalisation"

            abstract_model: Encoder_Decoder = nnx.eval_shape(lambda: Encoder_Decoder(nnx.Rngs(0)))
            graphdef, abstract_state = nnx.split(abstract_model)
            restored_state = checkpointer.restore(checkpoint_dir, abstract_state)
            model = nnx.merge(graphdef, restored_state)
            model.eval() # recursively sets deterministic=True and use_running_average=True

            @nnx.jit
            def apply(m: Encoder_Decoder, x):
                return m(x)

            obstacle_map_np = convert_bool_to_obstacle_map(self.walls)
            distance_map_np = generate_euclid_transform_map(obstacle_map_np)
            goal = gen_random_goal(obstacle_map_np)
            goal_map_np = generate_goal_map(obstacle_map_np, goal)

            final = np.stack([obstacle_map_np, distance_map_np, goal_map_np], axis=-1)

            self.heuristic_cache = apply(model, final)
            print("Heuristic map generated!")

        print(f"{node.state[0], node.state[1]} {self.heuristic_cache[node.state[0], node.state[1]] * LABEL_NORMALISER}")

        return self.heuristic_cache[node.state[0], node.state[1]] * LABEL_NORMALISER

    
    def astar_solve(self) -> None:
        start_node = Node(state=self.start, parent=None, action=None, depth=0)
        frontier = PriorityQueueFrontier(cost_fn=lambda n: n.depth + self.cnn_heuristic(n), initial_nodes=[start_node]) # decides order
        reached: dict[State, Node] = {start_node.state: start_node} # have I seen this state before, and was it via a better path?
        explored: set[State] = set() # prevents reexpansion for consistent heuristic

        while not frontier.empty():
            node = frontier.pop()
            self.num_explored += 1

            explored.add(node.state)
            
            for neighbour in self.neighbors(node):
                s = neighbour.state
                if s not in explored and (s not in reached or neighbour.depth < reached[s].depth):
                    if neighbour.state == self.goal:
                        self.solution = neighbour
                        return
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
