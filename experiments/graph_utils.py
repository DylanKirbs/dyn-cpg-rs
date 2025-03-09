from typing import List

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import tree_sitter_python
from tree_sitter import Language, Node, Parser
from tree_sitter import Range as tsRange
from tree_sitter import Tree

matplotlib.use("TkAgg")

PY_LANGUAGE = Language(tree_sitter_python.language())
PY_PARSER = Parser(PY_LANGUAGE)

LABELED_TYPES = [
    "string",
    "identifier",
    "integer",
    "float",
]


def draw_graph(graph: nx.Graph):
    """
    Draw the graph using matplotlib.

    :param graph: The graph to draw
    """

    pos = nx.multipartite_layout(graph, subset_key="layer", align="horizontal", scale=2)
    for k in pos:
        pos[k][-1] *= -1

    labels = nx.get_node_attributes(graph, "label")
    colors = nx.get_node_attributes(graph, "color")
    edge_colors = nx.get_edge_attributes(graph, "color")

    nx.draw(
        graph,
        pos,
        labels=labels,
        with_labels=True,
        node_color=colors.values(),
        edge_color=edge_colors.values(),
    )

    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.show()


def graph_to_tex(graph: nx.Graph) -> str:
    pos = nx.shell_layout(graph, scale=5)
    node_options = nx.get_node_attributes(graph, "color")
    tex = str(nx.to_latex_raw(graph, pos, node_options=node_options))  # type: ignore
    tex = tex.replace("_", "\\_")
    return tex


def _add_nodes(
    graph: nx.DiGraph, node: Node, changed_ranges: List[tsRange] = [], parent=None
):
    """
    Recursively add nodes and edges to the graph.

    :param graph: The graph to add nodes to
    :param node: The current node to add
    :param changed_ranges: The changed ranges in the new tree
    :param parent: The parent node (optional)
    """

    modified = False
    for range in changed_ranges:
        if range.start_byte <= node.start_byte <= range.end_byte:
            modified = True
            break

    label = node.type
    if node.type in LABELED_TYPES:
        label += (": " + node.text.decode()) if node.text else ""
    graph.add_node(
        node.id,
        label=label,
        layer=node.start_point.row,
        color="red" if modified else "blue",
    )

    if parent is not None:
        graph.add_edge(parent.id, node.id, label="AST", color="black")

    prev_child = None
    for child in node.children:
        _add_nodes(graph, child, changed_ranges, node)
        if prev_child is not None:
            graph.add_edge(prev_child.id, child.id, label="order", color="gray")
        prev_child = child


def tree_to_graph(tree: Tree) -> nx.DiGraph:
    """
    Convert a tree to a graph.

    :param tree: The tree to convert
    :return: The graph
    """

    G = nx.DiGraph()
    _add_nodes(G, tree.root_node)
    return G


def gen_highlighted_change_graph(old_tree: Tree, new_tree: Tree) -> nx.DiGraph:
    """
    Uses networkx to display the new tree with the changes highlighted.

    :param tree: The old tree
    :param new_tree: The new tree
    :return: The graph
    """

    G = nx.DiGraph()

    changed_ranges = old_tree.changed_ranges(new_tree)
    _add_nodes(G, new_tree.root_node, changed_ranges)

    return G
