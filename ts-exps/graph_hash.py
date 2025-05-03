from graph_utils import PY_PARSER, nx, tree_to_graph, draw_graph


class Sample:
    def __init__(self, code: str):
        self.code = code
        self.tree = PY_PARSER.parse(bytes(code, "utf8"))
        self.graph = tree_to_graph(self.tree)
        self.hash = nx.weisfeiler_lehman_graph_hash(self.graph, node_attr="label")

    def display(self):
        draw_graph(self.graph)

    def __hash__(self) -> int:
        return int(self.hash, 16)

    def __str__(self):
        return f"Code Sample: {self.hash[:6]}"


s = Sample("x = 1\ny = input()\na = x + y\nprint(a)")
s.display()

print(s)
