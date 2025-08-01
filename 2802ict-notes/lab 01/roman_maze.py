from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal
from typing import TypeAlias

Action: TypeAlias = Literal["u", "d", "l", "r"] | None
State: TypeAlias = tuple[int, int]

@dataclass
class Node:
    state: State
    parent: "Node | None"
    action: Action

@dataclass
class StackFrontier:
    frontier: list[Node] = field(default_factory=list)

    def add(self, node: Node):
        self.frontier.append(node)

    def extend(self, node: list[Node] | Iterable[Node]):
        self.frontier.extend(node)

    def remove(self, node: Node):
        self.frontier.remove(node)

    def empty(self):
        return len(self.frontier) == 0

    def __contains__(self, state: State) -> bool:
        return any(node.state == state for node in self.frontier)

    def pop(self) -> Node:
        return self.frontier.pop()

    def size(self) -> int:
        return len(self.frontier)


class QueueFrontier(StackFrontier):
    def pop(self) -> Node:
        return self.frontier.pop(0)


class Maze:
    def __init__(self):
        self.map: list[list[bool]] | None = None
        self.start: State = (0,0)
        self.goal: State = (0,0)

    def solve(self) -> Node | None:
        frontier = StackFrontier()
        explored: set[State] = set()
        start_node = Node(self.start, None, None)

        frontier.extend(self.explore(start_node))

        while frontier.size() > 0:
            cur_node = frontier.pop()

            explored.add(cur_node.state)

            # Goal check
            if cur_node.state == self.goal:
                return cur_node

            frontier.extend(filter(lambda k: k.state not in explored and k.state not in frontier, self.explore(cur_node)))

        return None

    def explore(self, node: Node) -> Iterable[Node]:

        col, row = node.state
        candidates = [
            Node((col - 1, row), node, "l"),
            Node((col + 1, row), node, "r"),
            Node((col, row - 1), node, "u"),
            Node((col, row + 1), node, "d"),
        ]

        check: Callable[[Node], bool] = lambda c: (
            self.map is not None
            and 0 <= c.state[0] < len(self.map[0])
            and 0 <= c.state[1] < len(self.map)
            and (not self.map[c.state[1]][c.state[0]])
        )

        return filter(check, candidates)

    def load_map(self, file: str):
        with open(file, "r") as map_file:
            lines = map_file.readlines()
            self.map = []
            for line_no, line in enumerate(lines):
                self.map.append([])
                for character_no, char in enumerate(line):
                    self.map[line_no].append(char == "#")
                    if char == "B":
                        self.start = (character_no, line_no)
                    if char == "A":
                        self.goal = (character_no, line_no)

    def print_map(self):
        if self.map == None:
            return
        print(f"goal={self.goal}, start={self.start}")
        for row in self.map:
            print("".join(map(lambda x: "█" if x else " ", row)))


if __name__ == "__main__":
    DemoMaze = Maze()
    DemoMaze.load_map("class notes/lab 01/maze1.txt")
    DemoMaze.print_map()

    sol = DemoMaze.solve()

    actions: list[Action] = []

    node = sol
    while node:
        actions.append(node.action)
        node = node.parent

    print(f"solution = {list(reversed(actions))}")
