import pytest
from crossword import *
from generate import *

@pytest.fixture
def crossword_0() -> CrosswordCreator:
    crossword = Crossword("data/structure0.txt", "data/words0.txt")
    creator = CrosswordCreator(crossword)
    return creator

def test_revise(crossword_0):
    var_1 = Variable(0, 0, 'down', 3)
    var_2 = Variable(0, 0, 'across', 3)
    crossword_0.crossword.overlaps = {(var_1, var_2): (0,0)}

    crossword_0.domains = {var_1: {"cat"}, var_2: {"cot"}}
    result = crossword_0.revise(var_1, var_2)
    assert result == False
    
    crossword_0.domains = {var_1: {"cat"}, var_2: {"bot"}}
    result = crossword_0.revise(var_1, var_2)
    assert result == True
    assert len(crossword_0.domains[var_1]) == 0