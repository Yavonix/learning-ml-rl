import polars as pl
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Union, cast

Status = Literal["unacc", "acc", "good", "vgood"]
    
CAR_COLUMNS = {
    "buying": pl.Enum(["low", "med", "high", "vhigh"]),
    "maint": pl.Enum(["low", "med", "high", "vhigh"]),
    "doors": pl.Enum(["2", "3", "4", "5more"]),
    "persons": pl.Enum(["2", "3", "4", "more"]),
    "lug_boot": pl.Enum(["small", "med", "big"]),
    "safety": pl.Enum(["low", "med", "high"]),
    "class": pl.Enum(["unacc", "acc", "good", "vgood"]),
}
OUTCOME_CLASS = "class"

@dataclass
class InteriorNode:
    attribute: str
    plurality_prediction: Status
    decision_tree: 'dict[str, Node]' = field(default_factory=dict)

@dataclass
class LeafNode:
    value: Status

type Node = Union[InteriorNode, LeafNode]

def entropy(examples: pl.DataFrame):
    outcomes = examples[OUTCOME_CLASS].unique()
    total = examples.height
    chances = list(map(lambda k: examples.filter(pl.col(OUTCOME_CLASS) == k).height / total, outcomes))
    return -sum(p * math.log(p, 2) for p in chances if p > 0)

def remainder(examples: pl.DataFrame, attribute: str):
    total = examples.height
    final = 0
    ## Weighted average of entropy when filtered by each attribute value
    for attr in examples[attribute].unique():
        subset_examples = examples.filter(pl.col(attribute) == attr)
        final += (subset_examples.height/total)*entropy(subset_examples)
    return final

def importance(examples: pl.DataFrame, attribute: str):
    ## The size of the reduction in entropy
    return entropy(examples) - remainder(examples, attribute)

def plurality_value(examples: pl.DataFrame) -> Status:
    mode_value = examples[OUTCOME_CLASS].mode().first()
    return cast(Status, mode_value) # polars return types make me sad

def ID3(examples: pl.DataFrame, attributes: set[str], parent_examples: pl.DataFrame) -> Node:
    if examples.height == 0:
        return LeafNode(plurality_value(parent_examples))
    elif examples[OUTCOME_CLASS].n_unique() == 1:
        return LeafNode(plurality_value(examples))
    elif len(attributes) == 0:
        return LeafNode(plurality_value(examples))
    else:
        ## Find best attribute
        decision_attribute = max(attributes, key=lambda x: importance(examples, x))
 
        ## Construct tree
        tree = InteriorNode(decision_attribute, plurality_value(examples))

        for value in examples[decision_attribute].unique():
            filtered_examples = examples.filter(pl.col(decision_attribute) == value)
            subtree = ID3(filtered_examples, attributes - {decision_attribute}, examples)
            tree.decision_tree[value] = subtree
        
        return tree

def classify(row: dict[str, Any], node: Node) -> Status:
    if isinstance(node, LeafNode):
        return node.value
    elif isinstance(node, InteriorNode):
        value = row[node.attribute]
        if value not in node.decision_tree: return node.plurality_prediction 
        else: return classify(row, node.decision_tree[value])

def pretty_print_tree(node: Node, indent: str = ""):
    if isinstance(node, LeafNode):
        print(f"{indent}Leaf: {node.value}")
    elif isinstance(node, InteriorNode):
        print(f"{indent}Attribute: {node.attribute}")
        for value, subtree in node.decision_tree.items():
            print(f"{indent}  Value: {value}")
            pretty_print_tree(subtree, indent + "    ")

def main():
    df_csv = pl.read_csv("./dt-car.csv", schema_overrides=CAR_COLUMNS)
    
    tree = ID3(df_csv, set(df_csv.columns) - {OUTCOME_CLASS}, df_csv)

    pretty_print_tree(tree)

if __name__ == "__main__":
    main()
