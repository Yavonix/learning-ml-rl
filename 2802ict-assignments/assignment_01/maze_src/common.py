import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal
from typing import TypeAlias, Any

Action: TypeAlias = Literal["u", "d", "l", "r"]
State: TypeAlias = tuple[int, int]

@dataclass
class Node:
    state: State
    parent: 'Node | None'
    action: Action | None
    depth: int = 0

    ## for the priority queue if two nodes have the same computed f cost
    def __lt__(self, other: 'Node') -> bool:
        return self.depth < other.depth

class Maze_Common():

    def __init__(self, filename):
        self.heuristic_cache: Any = None
        if filename:
            self.load_file(filename)

    def load_file(self, filename):
        # Read file and set height and width of maze
        with open(filename) as f:
            contents = f.read()

        # Validate start and goal
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Determine height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Keep track of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution: Node | None = None
        self.explored: set[State] = set()

    def reset_stats(self):
        self.num_explored = 0
        self.solution = None
        self.explored = set()
        self.heuristic_cache = None

    def print(self, skip_map_print:bool=False):
        actions, path = self.recover_path(self.solution) if self.solution is not None else [None, None]

        cost = len(actions) if actions else 0

        if actions: print(f"a. solution={''.join(actions)}")
        else: print("a. no solution found")

        print(f"b. cost={cost}")
        print(f"c. path_length={cost}")
        print(f"d. states_explored={self.num_explored}")
        print(f"e. original_maze:")
        if skip_map_print: print("skip_map_print active")
        else: self.print_maze(None)
        print(f"f. visualised_path:")
        if skip_map_print: print("skip_map_print active")
        else: self.print_maze(path)

        print()

    def print_maze(self, path: list[tuple[int,int]]|None):
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif path is not None and (i, j) in path:
                    print(f"{(path.index((i, j))+1)%10}", end="")
                else:
                    print(" ", end="")
            print()

    def neighbors(self, node: Node) -> Iterable[Node]:
        row, col = node.state

        candidates = [
            Node((row - 1, col), node, "u", depth=node.depth+1),
            Node((row + 1, col), node, "d", depth=node.depth+1),
            Node((row, col - 1), node, "l", depth=node.depth+1),
            Node((row, col + 1), node, "r", depth=node.depth+1),
        ]

        check: Callable[[Node], bool] = lambda c: (
            self.walls is not None and
            0 <= c.state[0] < self.height and
            0 <= c.state[1] < self.width and
            (not self.walls[c.state[0]][c.state[1]])
        )

        return filter(check, candidates)

    def recover_path(self, node: Node) -> tuple[list[Action], list[tuple[int,int]]]:
        actions: list[Action] = []
        cells: list[tuple[int, int]] = []
        while node.parent is not None:
            if node.action: actions.append(node.action)
            cells.append(node.state)
            node = node.parent
        actions.reverse()
        cells.reverse()
        return (actions, cells)

    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        path = self.recover_path(self.solution)[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solution
                elif path is not None and show_solution and (i, j) in path:
                    fill = (220, 235, 113)

                # Explored
                elif show_explored and self.explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )

        img.save(filename)

class NoSolutionError(Exception):
    pass

class DepthLimitReachedError(Exception):
    pass

class FrontierADT():    
    def empty(self):
        raise NotImplementedError
    
    def add(self, node: Node):
        raise NotImplementedError
    
    def extend(self, nodes: list[Node] | Iterable[Node]):
        raise NotImplementedError
    
    def remove(self, node: Node):
        raise NotImplementedError
    
    def pop(self) -> Node:
        raise NotImplementedError
    
    def size(self) -> int:
        raise NotImplementedError