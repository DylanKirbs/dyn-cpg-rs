from graph_utils import (
    PY_LANGUAGE,
    PY_PARSER,
    draw_graph,
    gen_highlighted_change_graph,
    nx,
    plt,
    tree_to_graph,
    tsRange,
)

c1 = """
def foo():
    x = source()
    if x < MAX:
        y = 2 * x
        sink(y)
"""

c2 = """
def foo():
    x = source()
    if x > MAX:
        y = 2 * x
        sink(y)
"""

orig_tree = PY_PARSER.parse(bytes(c1, "utf-8"))
new_tree = PY_PARSER.parse(bytes(c2, "utf-8"))


graph = gen_highlighted_change_graph(orig_tree, new_tree)
draw_graph(graph)
