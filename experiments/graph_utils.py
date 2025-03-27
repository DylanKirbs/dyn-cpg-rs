from collections import defaultdict, deque
from copy import deepcopy
from itertools import zip_longest
from typing import List

import matplotlib
import networkx as nx

from tree_sitter import Node, Query, Tree, Range as tsRange

matplotlib.use("TkAgg")


LABELED_TYPES = [
    "string",
    "identifier",
    "integer",
    "float",
]


class NodeProps:

    def __init__(self):
        self.is_root = False
        self.is_changed = False
        self.is_error = False
        self.has_changed_child = False

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


def ast_layout(G: nx.Graph, scale=1, center=None):
    """
    Position nodes of an AST in layers without edge intersections, following a partite scheme.

    The tree may contain edges that violate the tree structure, only edges named "AST (child)" are considered when constructing the layout.

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

    G = deepcopy(G)

    # remove edges without the AST child label
    edges = list(G.edges(data=True))
    for u, v, data in edges:
        if data.get("label") != "AST (child)":
            G.remove_edge(u, v)

    if not nx.is_tree(G):
        raise nx.NetworkXError("G is not a tree. AST layout requires a tree structure.")

    if len(G) == 0:
        return {}

    is_directed = G.is_directed()

    # Determine root node(s)
    if is_directed:
        roots = [n for n in G if G.in_degree(n) == 0]  # type: ignore
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
        neighbours = G.neighbors(node) if not is_directed else G.successors(node)  # type: ignore
        children = []
        for neighbour in neighbours:
            if (not is_directed and neighbour != parent) or is_directed:
                children.append(neighbour)
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


def draw_graph(graph: nx.Graph, ax=None, **kwargs):
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
        ax=ax,
        **kwargs,
    )

    edge_labels = nx.get_edge_attributes(graph, "label")
    bbox = {"boxstyle": "round", "ec": (1.0, 1.0, 1.0), "fc": (0.2, 0.2, 0.2)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, bbox=bbox)


def graph_to_tex(graph: nx.Graph) -> str:
    pos = ast_layout(graph)
    node_options = nx.get_node_attributes(
        graph, "label"
    )  # TODO: include color at some point
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
        # changed child: if the node is within a changed range
        if range.start_byte <= node.start_byte and range.end_byte >= node.end_byte:
            props.has_changed_child = True

        # changed self: if the node is the changed range
        if range.start_byte >= node.start_byte and range.end_byte <= node.end_byte:
            props.is_changed = True

    props.is_root = parent is None
    props.is_error = node.type == "ERROR"

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
            get_node_uid(parent),
            node_uid,
            label="AST (child)",
            color=(0, 0, 0, 0.5),
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
    old_tree: Tree,
    new_tree: Tree,
    query: Query | None = None,
) -> nx.DiGraph:
    """
    Uses networkx to display the new tree with the changes highlighted.

    Makes use of tree-sitter's changed_ranges method to highlight the changes.

    :param tree: The old tree
    :param new_tree: The new tree
    :param query: The granularity of the query (must be a valid tree-sitter query with the label 'highlight', e.g. "(block) @highlight")
    :return: The graph
    """

    changed_ranges = old_tree.changed_ranges(new_tree)

    if query is None:
        return highlighted_graph(new_tree, changed_ranges)

    # Highlight the changes in the query
    orig_captures = query.captures(old_tree.root_node)
    new_captures = query.captures(new_tree.root_node)

    for o_node, n_node in zip_longest(
        sorted(orig_captures.get("highlight", []), key=lambda x: x.start_byte),
        sorted(new_captures.get("highlight", []), key=lambda x: x.start_byte),
    ):
        if o_node is None or n_node is None:
            # Added or removed range
            continue

        if (
            o_node.start_byte != n_node.start_byte
            or n_node.has_changes
            or o_node.has_changes
            or o_node.text != n_node.text
        ):
            changed_ranges.append(
                tsRange(
                    n_node.start_point,
                    n_node.end_point,
                    n_node.start_byte,
                    n_node.end_byte,
                )
            )

    return highlighted_graph(new_tree, changed_ranges)


def highlighted_query_graph(tree: Tree, query: Query, labels: List[str]) -> nx.DiGraph:
    captures = query.captures(tree.root_node)
    matches = []
    for label in labels:
        matches.extend(captures.get(label, []))

    highlights: List[tsRange] = []
    for match in matches:
        highlights.append(
            tsRange(
                match.start_point,
                match.end_point,
                match.start_byte,
                match.end_byte,
            )
        )

    return highlighted_graph(tree, highlights)


def highlighted_graph(tree, highlight_ranges):
    G = nx.DiGraph()
    _add_nodes(G, tree.root_node, highlight_ranges)
    return G
