from graph_utils import PY_PARSER, PY_LANGUAGE, tree_to_graph, draw_graph, plt, tsRange


code = """
a: int: float: boolean: pigeon = 5
"""

tree = PY_PARSER.parse(bytes(code, "utf-8"))

leaf_query = PY_LANGUAGE.query("_ @highlight")

captures = leaf_query.captures(tree.root_node)
highlights = []
for node in captures.get("highlight", []):
    if node.child_count > 0:
        continue
    highlights.append(
        tsRange(
            node.start_point,
            node.end_point,
            node.start_byte,
            node.end_byte,
        )
    )

graph = tree_to_graph(tree, highlights)
draw_graph(graph)
plt.show()
