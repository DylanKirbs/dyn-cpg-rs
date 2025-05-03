from graph_utils import (
    PY_PARSER,
    draw_graph,
    gen_highlighted_change_graph,
    graph_to_tex,
)

# Parse the old and new code
old_tree = PY_PARSER.parse(bytes("a = 1", "utf8"))
new_tree = PY_PARSER.parse(bytes("a = 'test'", "utf8"))

graph = gen_highlighted_change_graph(old_tree, new_tree)

draw_graph(graph)
print(graph_to_tex(graph))
