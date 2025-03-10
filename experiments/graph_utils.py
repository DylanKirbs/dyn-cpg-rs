from itertools import zip_longest
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

from collections import defaultdict, deque

import networkx as nx
import numpy as np


class NodeProps:

    def __init__(self):
        self.is_root = False
        self.is_changed = False

    def get_prop_col(self, alpha=0.5):

        col_keys = [
            k
            for k, _ in self.__dict__.items()
            if k.startswith("is_") or k.startswith("has_")
        ]

        num_keys_per_channel = len(col_keys) // 3

        col = [0, 0, 0]
        for i, key in enumerate(col_keys):
            index = i % 3
            col[index] += 1 if getattr(self, key) else col[index]

        return [c / (num_keys_per_channel + 1) for c in col[:3]] + [alpha]


def get_node_uid(node):
    uid = f"{node.type}:{node.start_byte}:{node.end_byte}"
    if node.text:
        uid += f":{node.text.decode()}"
    return uid


def ast_layout(G, scale=1, center=None):
    """
    Position nodes of an AST in layers without edge intersections, following a partite scheme.

    Parameters
    ----------
    G : NetworkX graph
        A tree (directed or undirected) representing the AST.
    scale : float, optional
        Scale factor for positions. Default is 1.
    center : array-like or None, optional
        Coordinate pair around which to center the layout. Default is None.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Raises
    ------
    NetworkXError
        If G is not a tree or has multiple roots.
    """

    if not nx.is_tree(G):
        raise nx.NetworkXError("G is not a tree. AST layout requires a tree structure.")

    if len(G) == 0:
        return {}

    is_directed = G.is_directed()

    # Determine root node(s)
    if is_directed:
        roots = [n for n in G if G.in_degree(n) == 0]
    else:
        roots = [next(iter(G.nodes))]  # Choose arbitrary root for undirected

    if len(roots) != 1:
        raise nx.NetworkXError("The graph is not a single-rooted tree.")
    root = roots[0]

    # BFS to assign layers and order nodes
    layers = defaultdict(list)
    visited = {root: True}
    queue = deque([(root, None, 0)])
    layers[0].append(root)

    while queue:
        node, parent, depth = queue.popleft()
        neighbors = G.neighbors(node) if not is_directed else G.successors(node)
        children = []
        for neighbor in neighbors:
            if (not is_directed and neighbor != parent) or is_directed:
                children.append(neighbor)
        for child in children:
            if child not in visited:
                visited[child] = True
                layers[depth + 1].append(child)
                queue.append((child, node, depth + 1))

    # Prepare subset_key for multipartite_layout
    subset_key = {depth: layer for depth, layer in layers.items()}

    # Use multipartite_layout with vertical alignment
    pos = nx.multipartite_layout(
        G,
        subset_key=subset_key,  # type: ignore
        align="vertical",
        scale=scale,
        center=center,
    )

    for k in pos:
        pos[k][-1] *= -1

    return pos


def draw_graph(graph: nx.Graph):
    """
    Draw the graph using matplotlib.

    :param graph: The graph to draw
    """

    pos = ast_layout(graph)

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
    graph: nx.DiGraph,
    node: Node,
    changed_ranges: List[tsRange] = [],
    parent=None,
):
    """
    Recursively add nodes and edges to the graph.

    :param graph: The graph to add nodes to
    :param node: The current node to add
    :param changed_ranges: The changed ranges in the new tree
    :param parent: The parent node (optional)
    """

    props = NodeProps()

    for range in changed_ranges:
        if range.start_byte <= node.start_byte and range.end_byte >= node.end_byte:
            props.is_changed = True
            break

    props.is_root = parent is None

    label = node.type
    if node.type in LABELED_TYPES:
        label += (": " + node.text.decode()) if node.text else ""
    label += f"\n({node.start_byte}:{node.end_byte})"

    col = props.get_prop_col()
    node_uid = get_node_uid(node)

    graph.add_node(
        node_uid,
        label=label,
        color=col,
    )

    if parent is not None:
        graph.add_edge(
            get_node_uid(parent), node_uid, label="AST (child)", color=(0, 0, 0, 0.5)
        )

    for child in node.children:
        _add_nodes(graph, child, changed_ranges, node)


def tree_to_graph(tree: Tree) -> nx.DiGraph:
    """
    Convert a tree to a graph.

    :param tree: The tree to convert
    :return: The graph
    """

    G = nx.DiGraph()
    _add_nodes(G, tree.root_node)
    return G


def gen_highlighted_change_graph(
    old_tree: Tree, new_tree: Tree, q_gran: str = "block"
) -> nx.DiGraph:
    """
    Uses networkx to display the new tree with the changes highlighted.

    Makes use of tree-sitter's changed_ranges method to highlight the changes.
    Makes an additional query at the children of the specified granularity to highlight textual changes.

    :param tree: The old tree
    :param new_tree: The new tree
    :param q_gran: The granularity of the query (must be a valid tree-sitter query)
    :return: The graph
    """

    G = nx.DiGraph()

    changed_ranges = old_tree.changed_ranges(new_tree)

    query = PY_LANGUAGE.query(f"({q_gran} (_) @{q_gran})")
    orig_captures = query.captures(old_tree.root_node)
    new_captures = query.captures(new_tree.root_node)
    for orig_range, new_range in zip_longest(
        orig_captures.get(q_gran, []),
        new_captures.get(q_gran, []),
    ):
        if orig_range is None or new_range is None:
            # Added or removed range
            continue

        if orig_range.text != new_range.text:
            changed_ranges.append(
                tsRange(
                    new_range.start_point,
                    new_range.end_point,
                    new_range.start_byte,
                    new_range.end_byte,
                )
            )

    _add_nodes(G, new_tree.root_node, changed_ranges)

    return G
