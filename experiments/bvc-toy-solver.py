import networkx as nx
from collections import defaultdict
from typing import Callable, List, Dict, Set, Tuple, Any, TypeVar, Generic

V = TypeVar("V")  # Vertex state type
M = TypeVar("M")  # Message content type
E = TypeVar("E")  # Edge property type


class BatchVertexCentricSolver(Generic[V, M, E]):
    """
    Generic batch vertex-centric algorithm solver for NetworkX graphs.

    This solver is completely agnostic to the specific algorithm being implemented
    and relies on externally provided functions for initialization, aggregation,
    update, propagation, and convergence checking.
    """

    def __init__(
        self,
        graph: nx.Graph,
        init_func: Callable[[nx.Graph], Dict[Any, V]],
        aggregate_func: Callable[[List[M]], M],
        update_func: Callable[[V, M], V],
        propagate_func: Callable[[nx.Graph, V, M, E], List[Tuple[Any, M]]],
        edge_property_func: Callable[[nx.Graph, Any, Any], E] = (lambda g, s, d: None),
        max_iterations: int = 100,
        convergence_func: Callable[[Dict[Any, V], Dict[Any, V]], bool] | None = None,
        debug: bool = False,
    ):
        """
        Initialize the solver with algorithm-specific functions.

        Parameters:
        - graph: NetworkX graph
        - init_func: Function to initialize vertex states
        - aggregate_func: Function to combine incoming messages
        - update_func: Function to update vertex state based on aggregated messages
        - propagate_func: Function to generate new messages
        - edge_property_func: Function to extract edge properties
        - max_iterations: Maximum number of iterations to run
        - convergence_func: Function to check if algorithm has converged
        - debug: Whether to print debug information
        """
        self.graph = graph
        self.init_func = init_func
        self.aggregate_func = aggregate_func
        self.update_func = update_func
        self.propagate_func = propagate_func
        self.edge_property_func = edge_property_func
        self.max_iterations = max_iterations
        self.convergence_func = convergence_func
        self.debug = debug

    def run(self) -> Dict[Any, V]:
        """
        Run the batch vertex-centric algorithm.

        Returns:
        - Dictionary of final vertex states
        """
        # Initialize vertex states using the provided init function
        vertex_states = self.init_func(self.graph)

        # Store initial aggregated messages (None by default)
        aggregated_messages = {v: None for v in vertex_states}

        # Main iteration loop
        iteration = 0
        converged = False

        while iteration < self.max_iterations and not converged:
            iteration += 1
            if self.debug:
                print(f"Starting iteration {iteration}")

            # Message passing phase
            messages = defaultdict(list)

            # Each vertex propagates messages to its neighbors
            for vertex_id, vertex_state in vertex_states.items():
                # Get current aggregated message
                current_agg_msg = aggregated_messages[vertex_id]

                # Generate outgoing messages
                for target, msg in self.propagate_func(
                    self.graph,
                    vertex_state,
                    current_agg_msg,  # type: ignore
                    self.edge_property_func(self.graph, vertex_id, None),
                ):
                    if target in vertex_states:  # Ensure target exists
                        messages[target].append(msg)

            # Aggregation and update phase
            new_vertex_states = {}
            new_agg_messages = {}

            for vertex_id, vertex_state in vertex_states.items():
                # Aggregate incoming messages
                incoming_messages = messages.get(vertex_id, [])
                if incoming_messages:
                    aggregated_msg = self.aggregate_func(incoming_messages)
                else:
                    # Use algorithm-specific empty message (None is passed to let
                    # aggregate_func decide what to return for empty input)
                    aggregated_msg = (
                        self.aggregate_func([])
                        if hasattr(self.aggregate_func, "__code__")
                        and self.aggregate_func.__code__.co_argcount > 0
                        else None
                    )

                # Store aggregated message
                new_agg_messages[vertex_id] = aggregated_msg

                # Update vertex state
                new_state = self.update_func(vertex_state, aggregated_msg)  # type: ignore
                new_vertex_states[vertex_id] = new_state

            # Check for convergence
            if self.convergence_func:
                converged = self.convergence_func(vertex_states, new_vertex_states)

            # Update states for next iteration
            vertex_states = new_vertex_states
            aggregated_messages = new_agg_messages

            if self.debug:
                print(f"Completed iteration {iteration}, converged: {converged}")

        return vertex_states


# Example: PageRank Implementation using the generic solver
def pagerank_example():
    # Create example graph
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 1), (3, 5), (4, 5), (5, 1)])

    # Define algorithm-specific functions
    def init_pagerank(graph):
        """Initialize vertex states for PageRank"""
        num_nodes = len(graph.nodes())
        init_rank = 1.0 / num_nodes

        states = {}
        for node in graph.nodes():
            states[node] = {
                "id": node,
                "rank": init_rank,
                "out_degree": max(1, graph.out_degree(node)),  # Avoid division by zero
            }
        return states

    def aggregate_pagerank(messages):
        """Sum incoming PageRank contributions"""
        return sum(messages) if messages else 0.0

    def update_pagerank(vertex_state, aggregated_msg):
        """Apply PageRank formula with damping factor"""
        damping_factor = 0.85
        num_nodes = 5  # In a real implementation, this should be passed

        new_rank = damping_factor * aggregated_msg + (1 - damping_factor) / num_nodes

        new_state = vertex_state.copy()
        new_state["rank"] = new_rank
        return new_state

    def propagate_pagerank(graph, vertex_state, aggregated_msg, edge_prop):
        """Distribute PageRank to neighbors"""
        vertex_id = vertex_state["id"]
        current_rank = vertex_state["rank"]
        out_degree = vertex_state["out_degree"]

        # Get successors
        successors = list(graph.successors(vertex_id))

        if not successors:
            return []

        # Each successor gets current_rank / out_degree
        contribution = current_rank / out_degree
        return [(succ, contribution) for succ in successors]

    def check_pagerank_convergence(old_states, new_states, tolerance=1e-6):
        """Check if PageRank values have converged"""
        for node in old_states:
            old_rank = old_states[node].get("rank", 0)
            new_rank = new_states[node].get("rank", 0)
            if abs(old_rank - new_rank) > tolerance:
                return False
        return True

    # Create and run the solver
    solver = BatchVertexCentricSolver(
        graph=G,
        init_func=init_pagerank,
        aggregate_func=aggregate_pagerank,
        update_func=update_pagerank,
        propagate_func=propagate_pagerank,
        max_iterations=50,
        convergence_func=lambda old, new: check_pagerank_convergence(old, new, 1e-6),
        debug=True,
    )

    # Run the algorithm
    final_states = solver.run()

    # Print results
    print("\nFinal PageRank values:")
    for node, state in sorted(final_states.items()):
        print(f"Node {node}: {state['rank']:.6f}")

    # Verify sum of PageRank values
    total_rank = sum(state["rank"] for state in final_states.values())
    print(f"\nSum of all PageRank values: {total_rank:.6f}")

    # Compare with NetworkX implementation
    nx_pagerank = nx.pagerank(G, alpha=0.85)
    print("\nComparison with NetworkX PageRank:")
    for node, rank in sorted(nx_pagerank.items()):
        print(
            f"Node {node}: {rank:.6f} (our implementation: {final_states[node]['rank']:.6f})"
        )


# Example: Connected Components using the generic solver
def connected_components_example():
    # Create example undirected graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5), (6, 7), (7, 8)])

    # Define algorithm-specific functions
    def init_cc(graph):
        """Initialize each vertex with its own ID as component ID"""
        return {node: {"id": node, "component": node} for node in graph.nodes()}

    def aggregate_cc(messages):
        """Find minimum component ID from messages"""
        return min(messages) if messages else float("inf")

    def update_cc(vertex_state, aggregated_msg):
        """Update component ID to minimum of current and received"""
        min_component = (
            min(vertex_state["component"], aggregated_msg)
            if aggregated_msg != float("inf")
            else vertex_state["component"]
        )

        new_state = vertex_state.copy()
        new_state["component"] = min_component
        return new_state

    def propagate_cc(graph, vertex_state, aggregated_msg, edge_prop):
        """Propagate current component ID to all neighbors"""
        vertex_id = vertex_state["id"]
        component = vertex_state["component"]

        # For undirected graph, use neighbors
        neighbors = list(graph.neighbors(vertex_id))

        # Send current component ID to all neighbors
        return [(neighbor, component) for neighbor in neighbors]

    def check_cc_convergence(old_states, new_states):
        """Check if component IDs have stabilized"""
        for node in old_states:
            if old_states[node]["component"] != new_states[node]["component"]:
                return False
        return True

    # Create and run the solver
    solver = BatchVertexCentricSolver(
        graph=G,
        init_func=init_cc,
        aggregate_func=aggregate_cc,
        update_func=update_cc,
        propagate_func=propagate_cc,
        max_iterations=20,
        convergence_func=check_cc_convergence,
        debug=True,
    )

    # Run the algorithm
    final_states = solver.run()

    # Print results
    print("\nConnected Components:")
    components = {}
    for node, state in sorted(final_states.items()):
        component = state["component"]
        if component not in components:
            components[component] = []
        components[component].append(node)

    for i, (component, nodes) in enumerate(sorted(components.items())):
        print(f"Component {i+1}: {nodes} (ID: {component})")


if __name__ == "__main__":
    print("Running PageRank example:")
    pagerank_example()

    print("\n" + "=" * 50 + "\n")

    print("Running Connected Components example:")
    connected_components_example()
