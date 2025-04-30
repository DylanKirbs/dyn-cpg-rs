import networkx as nx
from collections import defaultdict
from typing import Callable, List, Tuple, Any, TypeVar, Generic
import copy

G = TypeVar("G")
V = TypeVar("V")
M = TypeVar("M")
E = TypeVar("E")


class BatchVertexCentricSolver(Generic[G, V, M, E]):
    """
    Generic batch vertex-centric algorithm solver for graphs.

    This solver follows the formula:
    m_v^i = H(M_v^{i-1})                    (aggregation)
    x_v^i = U(x_v^{i-1}, m_v^i)             (update)
    m_v^{i,w} = G(x_v^i, m_v^i, P_E(v,w))   (propagation to neighbors w)

    Where:
    - H is the aggregate function
    - U is the update function
    - G is the propagate function
    - PE is the edge property function

    Args:
        Generic (G, V, M, E): Type variables for graph, vertex state, message content, and edge property.
    """

    def __init__(
        self,
        graph: G,
        init_func: Callable[[G], None],
        aggregate_func: Callable[[List[M]], M],
        update_func: Callable[[G, Any, M], None],
        propagate_func: Callable[[G, Any, M, E], List[Tuple[Any, M]]],
        get_vertex_ids_func: Callable[[G], List[Any]],
        get_vertex_state_func: Callable[[G, Any], V],
        edge_property_func: Callable[[G, Any, Any], E] = (lambda g, s, d: None),
        max_iterations: int = 100,
        convergence_func: Callable[[G, G], bool] | None = None,
        debug: bool = False,
    ):
        """
        Initialize the solver with algorithm-specific functions.

        Args:
            graph (G): The graph to run the solver on
            init_func (Callable[[G], None]): Function to initialize vertex states directly in the graph
            aggregate_func (Callable[[List[message]], message]): Function to combine incoming messages (H)
            update_func (Callable[[G, vertex_id, message], None]): Function to update vertex state directly in the graph (U)
            propagate_func (Callable[[G, vertex_id, message, edge_property], List[Tuple[vertex_id, message]]]):
                Function to generate new messages for neighbors (G)
            get_vertex_ids_func (Callable[[G], List[vertex_id]]): Function to get vertex IDs from the graph
            get_vertex_state_func (Callable[[G, vertex_id], vertex_state]): Function to extract vertex state from graph
            edge_property_func (Callable[[G, vertex_id, edge_id], edge_property]): Function to extract edge properties (PE)
            max_iterations (int, optional): The maximum number of iterations to run. Defaults to 100.
            convergence_func (Callable[[G, G], bool], optional): Function to check if algorithm has converged by comparing graphs
            debug (bool, optional): Whether to print debug information. Defaults to False.
        """
        self.graph = graph
        self.init_func = init_func
        self.aggregate_func = aggregate_func
        self.update_func = update_func
        self.propagate_func = propagate_func
        self.edge_property_func = edge_property_func
        self.get_vertex_state_func = get_vertex_state_func
        self.get_vertex_ids_func = get_vertex_ids_func
        self.max_iterations = max_iterations
        self.convergence_func = convergence_func
        self.debug = debug

    def run(self) -> Tuple[G, bool]:
        """
        Run the vertex-centric algorithm on the graph.

        Returns:
            Tuple(G, bool): The final graph and a boolean indicating if the algorithm converged.
        """

        # Setup
        working_graph = copy.deepcopy(self.graph)
        self.init_func(working_graph)
        vertices = self.get_vertex_ids_func(working_graph)
        current_agg_messages = {v: None for v in vertices}
        iteration = 0
        converged = False

        # BVC algorithm
        while iteration < self.max_iterations and not converged:
            iteration += 1

            if self.debug:
                print(f"Starting iteration {iteration}")

            prev_graph = copy.deepcopy(working_graph) if self.convergence_func else None
            messages = defaultdict(list)

            for vertex_id in vertices:
                current_agg_msg = current_agg_messages[vertex_id]
                for target, msg in self.propagate_func(
                    working_graph,
                    vertex_id,
                    current_agg_msg,  # type: ignore
                    self.edge_property_func(working_graph, vertex_id, None),
                ):
                    if target in vertices:
                        messages[target].append(msg)
            new_agg_messages = {}

            for vertex_id in vertices:
                incoming_messages = messages.get(vertex_id, [])
                if incoming_messages:
                    aggregated_msg = self.aggregate_func(incoming_messages)
                else:
                    aggregated_msg = (
                        self.aggregate_func([])
                        if hasattr(self.aggregate_func, "__code__")
                        and self.aggregate_func.__code__.co_argcount > 0
                        else None
                    )
                new_agg_messages[vertex_id] = aggregated_msg
                self.update_func(working_graph, vertex_id, aggregated_msg)  # type: ignore

            current_agg_messages = new_agg_messages
            vertices = self.get_vertex_ids_func(working_graph)

            # Check for convergence
            if self.convergence_func and prev_graph:
                converged = self.convergence_func(prev_graph, working_graph)
            elif self.get_vertex_state_func:
                converged = (
                    all(
                        self.get_vertex_state_func(prev_graph, v)
                        == self.get_vertex_state_func(working_graph, v)
                        for v in self.get_vertex_ids_func(prev_graph)
                        if v in self.get_vertex_ids_func(working_graph)
                    )
                    if prev_graph
                    else False
                )
            else:
                converged = False

            if self.debug:
                print(f"Completed iteration {iteration}, converged: {converged}")

        if self.debug:
            print(
                f"Algorithm {'converged' if converged else 'terminated'} after {iteration} iterations."
            )
        return working_graph, converged


def pagerank_example():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 1), (3, 5), (4, 5), (5, 1)])
    for node in G.nodes():
        G.nodes[node]["data"] = {}

    def init_pagerank(graph):
        """Initialize vertex states for PageRank directly in the graph"""
        num_nodes = len(graph.nodes())
        init_rank = 1.0 / num_nodes

        for node in graph.nodes():
            graph.nodes[node]["data"] = {
                "rank": init_rank,
                "out_degree": max(1, graph.out_degree(node)),
            }

    def aggregate_pagerank(messages):
        """Sum incoming PageRank contributions"""
        return sum(messages) if messages else 0.0

    def update_pagerank(graph, vertex_id, aggregated_msg):
        """Apply PageRank formula with damping factor - modifies graph directly"""
        damping_factor = 0.85
        num_nodes = len(graph.nodes())

        vertex_data = graph.nodes[vertex_id]["data"]
        new_rank = damping_factor * aggregated_msg + (1 - damping_factor) / num_nodes
        vertex_data["rank"] = new_rank

    def propagate_pagerank(graph, vertex_id, aggregated_msg, edge_prop):
        """Distribute PageRank to neighbors"""
        vertex_data = graph.nodes[vertex_id]["data"]
        current_rank = vertex_data["rank"]
        out_degree = vertex_data["out_degree"]
        successors = list(graph.successors(vertex_id))

        if not successors:
            return []
        contribution = current_rank / out_degree
        return [(succ, contribution) for succ in successors]

    def get_vertex_ids(graph):
        """Get all vertex IDs from the graph"""
        return list(graph.nodes())

    def get_vertex_state(graph, vertex_id):
        """Extract vertex state from the graph"""
        return graph.nodes[vertex_id]["data"]

    def check_pagerank_convergence(prev_graph, curr_graph, tolerance=1e-6):
        """Check if PageRank values have converged by comparing graphs"""
        for node in prev_graph.nodes():
            if node not in curr_graph.nodes():
                continue

            old_rank = prev_graph.nodes[node]["data"].get("rank", 0)
            new_rank = curr_graph.nodes[node]["data"].get("rank", 0)

            if abs(old_rank - new_rank) > tolerance:
                return False
        return True

    solver = BatchVertexCentricSolver(
        graph=G,
        init_func=init_pagerank,
        aggregate_func=aggregate_pagerank,
        update_func=update_pagerank,
        propagate_func=propagate_pagerank,
        get_vertex_ids_func=get_vertex_ids,
        get_vertex_state_func=get_vertex_state,
        max_iterations=50,
        convergence_func=lambda old, new: check_pagerank_convergence(old, new, 1e-6),
        debug=True,
    )

    final_graph, _ = solver.run()

    print("\nFinal PageRank values:")
    for node in sorted(final_graph.nodes()):
        rank = final_graph.nodes[node]["data"]["rank"]
        print(f"Node {node}: {rank:.6f}")

    total_rank = sum(
        final_graph.nodes[node]["data"]["rank"] for node in final_graph.nodes()
    )
    print(f"\nSum of all PageRank values: {total_rank:.6f}")

    nx_pagerank = nx.pagerank(G)
    print("\nComparison with NetworkX PageRank:")
    for node, rank in sorted(nx_pagerank.items()):
        our_rank = final_graph.nodes[node]["data"]["rank"]
        print(f"Node {node}: {rank:.6f} (our implementation: {our_rank:.6f})")


def connected_components_example():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5), (6, 7), (7, 8)])
    for node in G.nodes():
        G.nodes[node]["data"] = {}

    def init_cc(graph):
        """Initialize each vertex with its own ID as component ID - directly in the graph"""
        for node in graph.nodes():
            graph.nodes[node]["data"] = {"component": node}

    def aggregate_cc(messages):
        """Find minimum component ID from messages"""
        return min(messages) if messages else float("inf")

    def update_cc(graph, vertex_id, aggregated_msg):
        """Update component ID to minimum of current and received - modifies graph directly"""
        vertex_data = graph.nodes[vertex_id]["data"]
        current_component = vertex_data["component"]
        if aggregated_msg != float("inf"):
            vertex_data["component"] = min(current_component, aggregated_msg)

    def propagate_cc(graph, vertex_id, aggregated_msg, edge_prop):
        """Propagate current component ID to all neighbors"""
        component = graph.nodes[vertex_id]["data"]["component"]
        neighbors = list(graph.neighbors(vertex_id))
        return [(neighbor, component) for neighbor in neighbors]

    def get_vertex_ids(graph):
        """Get all vertex IDs from the graph"""
        return list(graph.nodes())

    def get_vertex_state(graph, vertex_id):
        """Extract vertex state from the graph"""
        return graph.nodes[vertex_id]["data"]

    def check_cc_convergence(prev_graph, curr_graph):
        """Check if component IDs have stabilized by comparing graphs"""
        for node in prev_graph.nodes():
            if node not in curr_graph.nodes():
                continue

            prev_component = prev_graph.nodes[node]["data"]["component"]
            curr_component = curr_graph.nodes[node]["data"]["component"]

            if prev_component != curr_component:
                return False
        return True

    solver = BatchVertexCentricSolver(
        graph=G,
        init_func=init_cc,
        aggregate_func=aggregate_cc,
        update_func=update_cc,
        propagate_func=propagate_cc,
        get_vertex_ids_func=get_vertex_ids,
        get_vertex_state_func=get_vertex_state,
        max_iterations=20,
        convergence_func=check_cc_convergence,
        debug=True,
    )
    final_graph, _ = solver.run()

    print("\nConnected Components:")
    components = {}
    for node in sorted(final_graph.nodes()):
        component = final_graph.nodes[node]["data"]["component"]
        if component not in components:
            components[component] = []
        components[component].append(node)

    for i, (component, nodes) in enumerate(sorted(components.items())):
        print(f"Component {i+1}: {nodes} (ID: {component})")

    nx_components = list(nx.connected_components(G))
    print("\nComparison with NetworkX Connected Components:")
    for i, component in enumerate(nx_components):
        print(f"Component {i+1}: {sorted(component)}")


if __name__ == "__main__":
    print("Running PageRank example:")
    pagerank_example()

    print("\n" + "=" * 50 + "\n")

    print("Running Connected Components example:")
    connected_components_example()
