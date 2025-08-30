import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Any
import heapq
import time

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
from jax import numpy as jnp
import jax
from pathlib import Path

class StackFrontier(FrontierADT):
    def __init__(self, initial_nodes: list[Node] | None = None):
        if initial_nodes:
            self.frontier = initial_nodes
        else:
            self.frontier = []

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
    def __init__(self, cost_fn: Callable[[Node], Any], initial_nodes: list[Node] | None = None):
        self.frontier: list[tuple[Any,Node]] = []
        self.score = cost_fn
        if initial_nodes:
            for node in initial_nodes:
                heapq.heappush(self.frontier, (self.score(node), node))
    
    def empty(self):
        return len(self.frontier) == 0
    
    def add(self, node):
        heapq.heappush(self.frontier, (self.score(node), node))

    def extend(self, nodes):
        for node in nodes:
            heapq.heappush(self.frontier, (self.score(node), node))

    def remove(self, node):
        self.frontier = [t for t in self.frontier if t[1] is not node]
        heapq.heapify(self.frontier)

    def pop(self):
        return heapq.heappop(self.frontier)[1]
    
    def size(self):
        return len(self.frontier)
    
    def is_best(self, node: Node):
        smallest = heapq.nsmallest(1, self.frontier)[0]
        return self.score(node) <= smallest[0]


class Maze(Maze_Common):

    def dfs_solve(self, max_depth=float("inf")) -> None:
        """Depth-limited depth-first search"""

        start_node = Node(state=self.start, parent=None, action=None, depth=0)
        frontier = StackFrontier(initial_nodes=[start_node])
        cutoff_occurred = False

        while not frontier.empty():
            node = frontier.pop()
            self.num_explored += 1

            if node.state == self.goal:
                self.solution = node
                return

            self.explored.add(node.state) # just for image rendering

            for neighbour in self.neighbors(node):
                if neighbour.depth > max_depth:
                    cutoff_occurred = True
                    continue

                anc = node
                in_ancestors = False
                while anc is not None:
                    if anc.state == neighbour.state:
                        in_ancestors = True
                        break
                    anc = anc.parent

                if not in_ancestors:
                    frontier.add(neighbour)

        if cutoff_occurred:
            raise DepthLimitReachedError
        else:
            raise NoSolutionError

    def ids_solve(self, max_depth:int = 10000) -> None:
        """Iterative depth-first search"""

        depth = 1

        while depth <= max_depth:
            try:
                # print(f"trying {depth}")
                self.dfs_solve(max_depth=depth)
                return
            except DepthLimitReachedError:
                depth += 1
            except Exception as e:
                raise e

        raise DepthLimitReachedError

    
    
    def load_model(self, heuristic_weighting=10):
        """Load the CNN into memory"""
        checkpointer = ocp.StandardCheckpointer()
        checkpoint_dir = "./cnn_heuristic/checkpoints/save-no-normalisation-2100"
        full_checkpoint_path = Path(checkpoint_dir).resolve()

        abstract_model: Encoder_Decoder = nnx.eval_shape(lambda: Encoder_Decoder(nnx.Rngs(0)))
        graphdef, abstract_state = nnx.split(abstract_model)
        restored_state = checkpointer.restore(full_checkpoint_path, abstract_state)
        model = nnx.merge(graphdef, restored_state)
        model.eval() # recursively sets deterministic=True and use_running_average=True

        @nnx.jit
        def apply(m: Encoder_Decoder, x):
            return m(x)
        
        self.cnn_apply = lambda map: apply(model, map)

        # We run the model once with an empty input to trigger jax JIT compilation
        self.cnn_apply(jnp.zeros((224,224,3)))

        self.heuristic_weighting = heuristic_weighting
    
    def manhattan_heuristic(self, node: Node) -> int:
        return abs(self.goal[0]-node.state[0]) + abs(self.goal[1]-node.state[1])

    def cnn_heuristic(self, node: Node) -> int:
        assert self.width == self.height == 224

        if self.heuristic_cache is None:
            print("Generating heuristic map...")

            obstacle_map_np = convert_bool_to_obstacle_map(self.walls)
            distance_map_np = generate_euclid_transform_map(obstacle_map_np)
            # goal = gen_random_goal(obstacle_map_np)
            print(f"Using goal {self.goal}")
            goal_map_np = generate_goal_map(obstacle_map_np, self.goal)

            final = np.stack([obstacle_map_np, distance_map_np, goal_map_np], axis=-1)
            
            ## Run the model then transfer returned heuristic map from accelerator (if present) to CPU 
            self.heuristic_cache = jax.device_get(self.cnn_apply(final))
            print("Heuristic map generated!")

        # print(f"{node.state[0], node.state[1]} {self.heuristic_cache[node.state[0], node.state[1]] * LABEL_NORMALISER}")

        # technically a weighted A* search because of the * 10
        return self.heuristic_cache[node.state[0], node.state[1]] * LABEL_NORMALISER * self.heuristic_weighting


    def astar_solve_consistent(self) -> None:
        start_node = Node(state=self.start, parent=None, action=None, depth=0)
        frontier = PriorityQueueFrontier(cost_fn=lambda n: (n.depth + self.manhattan_heuristic(n),-n.depth), initial_nodes=[start_node]) # decides order
        reached: dict[State, Node] = {start_node.state: start_node} # have I seen this state before, and was it via a better path?
        explored: set[State] = set() # prevents reexpansion for consistent heuristic

        while not frontier.empty():
            node = frontier.pop()
            self.num_explored += 1

            self.explored.add(node.state) # just for the image rendering
            explored.add(node.state)
            
            for neighbour in self.neighbors(node):
                s = neighbour.state
                if s not in explored and (s not in reached or neighbour.depth < reached[s].depth):
                    if neighbour.state == self.goal:
                        self.solution = neighbour
                        return
                    reached[s] = neighbour
                    frontier.add(neighbour)

    def astar_solve(self) -> None:
        start_node = Node(state=self.start, parent=None, action=None, depth=0)
        heuristic_fn = lambda n: (n.depth + self.cnn_heuristic(n), -n.depth)
        # heuristic_fn:Callable[[Node], int] = lambda n: n.depth + self.manhattan_heuristic(n)
        frontier = PriorityQueueFrontier(cost_fn=heuristic_fn, initial_nodes=[start_node]) # decides order
        reached: dict[State, Node] = {start_node.state: start_node} # have I seen this state before, and was it via a better path?

        solution:Node|None = None

        while not frontier.empty():
            node = frontier.pop()
            self.num_explored += 1

            self.explored.add(node.state) # just for the image rendering

            if node.state == self.goal:
                if not solution or node.depth < solution.depth:
                    solution = node
            
            if solution and frontier.is_best(solution):
                self.solution=solution
                return
            
            for neighbour in self.neighbors(node):
                s = neighbour.state
                if s not in reached or neighbour.depth < reached[s].depth:
                    reached[s] = neighbour
                    frontier.add(neighbour)


if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze.txt")

if __name__ == "__main__":

    m = Maze(sys.argv[1])

    m.reset_stats()
    try:
        # m.astar_solve()
        m.ids_solve()
    except NoSolutionError:
        pass
    except Exception as e:
        raise e
    m.print()

    # print("Solution:\n\n")
    # m.output_image("maze.png", show_explored=True)
