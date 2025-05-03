"""
Proof of concept for incremental CPG construction.

If the incremental construction works here, it can be ported to C.
"""

import hashlib
from difflib import SequenceMatcher
from typing import Dict, List, Literal, Tuple
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import tree_sitter as ts
import tree_sitter_c as c_parser
import tree_sitter_python as py_parser
import utils

from concurrent.futures import ThreadPoolExecutor

matplotlib.use("TkAgg")  # Use TkAgg backend for matplotlib

# ----- Types ----- #

Diff = List[Tuple[Literal["replace", "delete", "insert"], int, int, int, int]]
"""A list of differences between text snippets.
Each difference is represented as a tuple containing:
- Operation type ('replace', 'delete', 'insert')
- Start index in the old text
- End index in the old text
- Start index in the new text
- End index in the new text
"""

# ----- Constants ----- #

LANGS: Dict[str, ts.Language] = {
    "c": ts.Language(c_parser.language()),
    "py": ts.Language(py_parser.language()),
}

TYPE_MAP: Dict[str, str] = {
    # TODO populate this with grammar bindings for all (supported) languages
    "if_statement": "if_predicate",
}

# ----- Functions ----- #


def parse_code(code: str, language: str, old_tree: ts.Tree | None = None) -> ts.Tree:
    """
    Parse the given code using the specified language parser.

    Args:
        code (str): The code to parse.
        language (str): The programming language of the code.
        old_tree (Tree, optional): The old parse tree to incrementally update. Defaults to None.

    Raises:
        ValueError: If the language is not supported.

    Returns:
        Tree: The parse tree.
    """

    lang = LANGS.get(language)
    if lang is None:
        raise ValueError(
            f"Unsupported language: {language}. Use {', '.join(sorted(LANGS.keys()))}."
        )

    parser = ts.Parser(language=lang)
    if old_tree is None:
        return parser.parse(bytes(code, "utf8"))

    return parser.parse(bytes(code, "utf8"), old_tree=old_tree)


def get_diff(old_code: str, new_code: str) -> Diff:
    """
    Get the diff between two code snippets.

    Args:
        old_code (str): The original code.
        new_code (str): The modified code.

    Returns:
        Diff: A list of tuples representing the differences.
            Each tuple contains the operation type ('replace', 'delete', 'insert'),
            start and end indices for both old and new code.
    """

    # Use SequenceMatcher to find the differences
    seq_matcher = SequenceMatcher(None, old_code, new_code)
    diff = list(filter(lambda op: op[0] != "equal", seq_matcher.get_opcodes()))

    return diff  # type: ignore[no-untyped-return]


def apply_diff(old_tree: ts.Tree, diff: Diff) -> None:
    """
    Apply the diff to the old parse tree to update it incrementally.

    Args:
        old_tree (Tree): The old parse tree.
        diff (Diff): The diff to apply.
    """

    for d in diff:
        old_tree.edit(
            start_byte=d[1],
            old_end_byte=d[2],
            new_end_byte=d[4],
            start_point=(0, 0),
            old_end_point=(0, 0),
            new_end_point=(0, 0),
        )


def agnostify(tree: ts.Tree) -> nx.DiGraph:
    """
    Convert the parse tree to a language agnostic graph representation.

    Args:
        tree (Tree): The parse tree.

    Returns:
        DiGraph: The graph representation of the parse tree.
    """

    # TODO this can be worked on to make it more language agnostic
    # but that's not the primary focus at the moment

    graph = nx.DiGraph()

    node_id_cache = {}

    def compute_id(node: ts.Node) -> str:
        # Check if we've already computed this node's ID
        if node.id in node_id_cache:
            return node_id_cache[node.id]

        if node.child_count == 0:  # Leaf node
            node_id = f"{TYPE_MAP.get(node.type, node.type)}-{node.start_byte}"
        else:  # Non-leaf node
            child_ids = [compute_id(child) for child in node.children]
            node_id = f"{TYPE_MAP.get(node.type, node.type)}-{'-'.join(child_ids)}"

        # Hash and cache the result
        hashed_id = hashlib.sha256(node_id.encode()).hexdigest()
        node_id_cache[node.id] = hashed_id
        return hashed_id

    def traverse(node: ts.Node, ordering=0) -> str:
        node_id = compute_id(node)
        graph.add_node(
            node_id,
            type=TYPE_MAP.get(node.type, node.type),
            code=(
                node.text.decode("utf-8")
                if node.text and node.child_count == 0
                else None
            ),
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            order=ordering,
        )
        for ordering, child in enumerate(node.children):
            child_id = traverse(child, ordering)
            graph.add_edge(node_id, child_id, label="AST")
        return node_id

    traverse(tree.root_node)
    return graph


def gpgify(graph: nx.DiGraph, node_id: str) -> List[str]:
    """
    Updates the descendants of a node with their CPG attributes.
    And marks them to be (re)processed.

    Args:
        graph (DiGraph): The graph representation of the parse tree.
        node_id (str): The ID of the node to process.

    Returns:
        List[str]: A list of node IDs that were marked for reprocessing.
    """

    print(f"DEBUG: Processing node {graph.nodes[node_id]}")

    # control flow (nodes with STMT & PRED, edges with e, true, false)
    # program dependence (same nodes as control flow, edges with data, control)
    if "statement" in graph.nodes[node_id]["type"]:
        # epsilon edge to the next statement or predicate (i.e. parent node's child with order + 1)
        ...
        # if this is a declaration, link a data edge to all statements that refer to it
        ...
    elif "predicate" in graph.nodes[node_id]["type"]:
        # true edge to first statement in the block
        # false edge to the first statement after the block
        ...
        # link a true control flow edge all statements in the block
        # link a false control flow edge all statements in the "else" block if any
        ...

    return []


# ----- Main ----- #


def main():

    old_code = """
    int main() {
        if (1 < 2) {
            printf("Hello, world!");
        } else {
            printf("Goodbye, world!");
        }
        return 0;
    }
    """

    new_code = """
    int main() {
        if (1 > 2) {
            printf("Hello, world!");
        } else {
            printf("Goodbye, world!");
        }
        printf("This is a new line.");
        return 0;
    }
    """

    old_tree = parse_code(old_code, "c")
    diff = get_diff(old_code, new_code)
    apply_diff(old_tree, diff)
    new_tree = parse_code(new_code, "c", old_tree=old_tree)

    changed_ranges = old_tree.changed_ranges(new_tree)

    for r in changed_ranges:
        print(
            f"DEBUG: Changed range: {r.start_byte} -> {r.end_byte} '{new_code[r.start_byte:r.end_byte+1].strip()}'"
        )

    graph = agnostify(new_tree)

    # in parallel, we update the graph with the CPG attributes
    with ThreadPoolExecutor() as executor:
        futures = []
        for node in graph.nodes:
            futures.append(executor.submit(gpgify, graph, node))
        for future in futures:
            re_proc = future.result()
            for node in re_proc:
                futures.append(executor.submit(gpgify, graph, node))

    # draw the graph
    plt.figure(figsize=(12, 8))
    pos = utils.ast_layout(graph, tree_attr=("label", "AST"))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        labels=nx.get_node_attributes(graph, "type"),
        node_size=700,
        node_color="lightblue",
        ax=plt.gca(),
    )
    plt.title("Language-Agnostic AST")
    # plt.show()


if __name__ == "__main__":
    main()
