import networkx as nx
from copy import deepcopy
from collections import defaultdict, deque


def ast_layout(G: nx.Graph, scale=1, center=None, tree_attr=("label", "AST Edge")):
    """
    Generate positions for nodes in a tree-like structure using a layout algorithm.

    Args:
        G (nx.Graph): The input graph representing the tree structure.
        scale (int, optional): The scale of the graph. Defaults to 1.
        center (tuple, optional): The center of the graph. Defaults to None.
        tree_attr (tuple, optional): The attribute to filter the edged on to form the tree. Defaults to ("label", "AST Edge").

    Raises:
        NetworkXError: If the graph is not a single-rooted tree.

    Returns:
        dict: A dictionary of node positions.
    """

    G = deepcopy(G)

    # Filter edges based on the specified tree attribute
    edges = list(G.edges(data=True))
    for u, v, data in edges:
        if data.get(tree_attr[0]) != tree_attr[1]:
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

    # Prepare subset_key for multipartite_layout (old way)
    # subset_key = {depth: layer for depth, layer in layers.items()}

    # Assign each node an attr "depth" based on its layer
    for depth, layer in layers.items():
        for node in layer:
            G.nodes[node]["depth"] = depth

    # Use multipartite_layout with vertical alignment
    pos = nx.multipartite_layout(
        G,
        subset_key="depth",
        align="vertical",
        scale=scale,
        center=center,
    )

    for k in pos:
        pos[k][-1] *= -1

    return pos
