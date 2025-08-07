import pytest

from crossword import *
from generate import *

structures  = ["data/structure0.txt", "data/structure1.txt", "data/structure2.txt"]
words = ["data/words0.txt", "data/words1.txt", "data/words2.txt"]
valid_crossword_params = list(zip(structures, words))

@pytest.fixture
def simple_crosswords() -> CrosswordCreator:
    crossword = Crossword(*valid_crossword_params[0])
    creator = CrosswordCreator(crossword)
    return creator

def _make_id(param):
    struct, wordlist = param
    s = struct.split("/")[-1].replace(".txt","")
    w = wordlist.split("/")[-1].replace(".txt","")
    return f"{s}__{w}"

@pytest.fixture(
    params=valid_crossword_params,
    ids=_make_id
)
def valid_crosswords(request) -> CrosswordCreator:
    structure_path, words_path = request.param
    crossword = Crossword(structure_path, words_path)
    creator = CrosswordCreator(crossword)
    return creator

def test_revise(simple_crosswords):
    var_1 = Variable(0, 0, 'down', 3)
    var_2 = Variable(0, 0, 'across', 3)
    simple_crosswords.crossword.overlaps = {(var_1, var_2): (0,0)}

    simple_crosswords.domains = {var_1: {"cat"}, var_2: {"cot"}}
    result = simple_crosswords.revise(var_1, var_2)
    assert result == False
    
    simple_crosswords.domains = {var_1: {"cat"}, var_2: {"bot"}}
    result = simple_crosswords.revise(var_1, var_2)
    assert result == True
    assert len(simple_crosswords.domains[var_1]) == 0

def test_valid_solve(valid_crosswords, benchmark):
    sol = benchmark(valid_crosswords.solve)
    assert sol is not None