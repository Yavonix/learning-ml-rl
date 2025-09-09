from functools import reduce
from itertools import product
import sys
from typing import Callable
from IPython.display import display

from crossword import *
from crossword_partly_filled import CrosswordPartlyFilled

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword: Crossword = crossword
        self.domains: dict[Variable, set[str]] = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename=""):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font) # type: ignore
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font # type: ignore
                        )
        if filename:
            img.save(filename)
        else:
            display(img)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """

        ## We check for two unary constraints:
        ##   1. That length is correct
        ##   2. That the variable fits any prefilled characters (for the special features/extensions part of the assignment)
        is_valid: Callable[[str,str],bool] = lambda candidate, mask: len(candidate) == len(mask) and all(c == m or m == '_' for c, m in zip(candidate.lower(), mask.lower()))

        self.domains = {
            variable: {k for k in domain if is_valid(k, variable.mask)}
            for variable, domain in self.domains.items()
        }

    def revise(self, x: Variable, y: Variable):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        if x == y: return False

        constraint = self.crossword.overlaps[(x, y)]
        if constraint == None: return False

        revision_made = False

        for x_value in self.domains[x].copy():
            value_found = False

            for y_value in self.domains[y]:
                if x_value[constraint[0]] == y_value[constraint[1]]:
                    value_found = True
                    break

            if not value_found:
                self.domains[x].remove(x_value)
                revision_made = True

        return revision_made

    def ac3(self, arcs: list[tuple[Variable, Variable]] | None = None) -> bool:
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        if not arcs: arcs = [(x, y) for (x, y), o in self.crossword.overlaps.items() if o]

        while arcs:
            arc = arcs.pop(0)
            if self.revise(arc[0], arc[1]):
                if len(self.domains[arc[0]]) == 0: return False # all remaining nodes gone
                arcs.extend([(neighbour, arc[0]) for neighbour in self.crossword.neighbors(arc[0])])

        return True
    
    def assignment_complete(self, assignment: dict[Variable, str]) -> bool:
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """

        return len(assignment) == len(self.crossword.variables)

    def consistent(self, assignment: dict[Variable, str]) -> bool:
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        # enforce uniqueness
        if len(assignment.values()) != len(set(assignment.values())): return False

        # enfore unary constraints
        for var, value in assignment.items():
            if value not in self.domains[var]: return False

        # enforce binary constraints
        for (x, y), constraint in self.crossword.overlaps.items():
            if constraint is None: continue
            if (x in assignment and
                y in assignment and 
                assignment[x][constraint[0]] != assignment[y][constraint[1]]):
                return False

        return True

    def order_domain_values(self, var: Variable, assignment: dict[Variable, str]) -> list[str]:
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        neighbours = self.crossword.neighbors(var)

        # we only need to check neighbours that do not yet have assignments
        unassigned_neighbours = neighbours - set(assignment.keys())

        ranking: dict[str, int] = {value: 0 for value in self.domains[var]}

        for neighbour in unassigned_neighbours:
            constraint = self.crossword.overlaps[(var, neighbour)]
            if not constraint: continue
            for (value, neighbour_value) in product(self.domains[var], self.domains[neighbour]):
                if value[constraint[0]] != neighbour_value[constraint[1]]: ranking[value] += 1

        ordered_ranking = sorted(ranking, key=lambda k: ranking[k])

        return ordered_ranking

    def select_unassigned_variable(self, assignment: dict[Variable, str]) -> Variable:
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """

        # Uses 2 heuristics for variable selection:
        ## Minimum remaining values: choose the variable with the fewest choices
        ## Most costraining values: choose the variable with the most constraints

        unassigned = self.crossword.variables - set(assignment.keys())

        def composite_key(var: Variable) -> tuple[int, int]:
            key1 = len(self.domains[var]) # Minimum remaining values: choose the variable with the fewest choices
            key2 = -len(self.crossword.neighbors(var)) # Most costraining values: choose the variable with the most constraints
            return (key1, key2)

        return min(unassigned, key=composite_key)

    def backtrack(self, assignment: dict[Variable, str]) -> dict[Variable, str] | None:
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a comple to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        if self.assignment_complete(assignment): return assignment

        var = self.select_unassigned_variable(assignment)
        
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result: return result
            del assignment[var]

        return None

def main():
    print("entry")
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = CrosswordPartlyFilled(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)

if __name__ == "__main__":
    main()