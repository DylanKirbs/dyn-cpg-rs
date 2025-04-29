import networkx as nx
from collections import defaultdict


def batch_vertex_centric_solver(
    G,
    aggregate_func,
    update_func,
    propagate_func,
    max_iterations=100,
    convergence_threshold=None,
):
    """
    Simple batch vertex-centric algorithm solver for NetworkX graphs.

    Parameters:
    - G: NetworkX graph
    - aggregate_func: Function to combine incoming messages
    - update_func: Function to update vertex state based on aggregated messages
    - propagate_func: Function to generate new messages
    - max_iterations: Maximum number of iterations to run
    - convergence_threshold: Optional convergence check function

    Returns:
    - Dictionary of vertex states after algorithm completion
    """
    # Initialize vertex states
    vertex_states = {node: {"id": node, "predecessors": set()} for node in G.nodes()}

    # Main iteration loop
    iteration = 0
    converged = False

    while iteration < max_iterations and not converged:
        iteration += 1
        print(f"Starting iteration {iteration}")

        # Message passing phase
        messages = defaultdict(list)

        # Each vertex propagates messages to its neighbors
        for vertex_id, vertex_state in vertex_states.items():
            outgoing_messages = propagate_func(
                G, vertex_state, vertex_state.get("predecessors", set()), None
            )  # Edge property is None for simplicity

            # Send messages to recipients
            for recipient, message in outgoing_messages:
                if recipient in vertex_states:  # Ensure recipient exists
                    messages[recipient].append(message)

        # Aggregation and update phase
        new_vertex_states = {}
        for vertex_id, vertex_state in vertex_states.items():
            # Aggregate incoming messages
            incoming_messages = messages.get(vertex_id, [])
            if incoming_messages:
                aggregated_msg = aggregate_func(G, incoming_messages)
            else:
                aggregated_msg = set()  # Empty set if no messages

            # Update vertex state
            new_state = update_func(G, vertex_state.copy(), aggregated_msg)
            new_vertex_states[vertex_id] = new_state

        # Check for convergence
        if convergence_threshold:
            converged = convergence_threshold(vertex_states, new_vertex_states)

        # Update states for next iteration
        vertex_states = new_vertex_states

        # Print some debugging info
        print(f"Completed iteration {iteration}, converged: {converged}")

    return vertex_states


# Example usage:
def create_example_graph():
    """Create a simple directed graph for testing"""
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])
    return G


# Example convergence check function
def check_convergence(old_states, new_states):
    """Check if the algorithm has converged by comparing states"""
    for node in old_states:
        old_preds = old_states[node].get("predecessors", set())
        new_preds = new_states[node].get("predecessors", set())
        if old_preds != new_preds:
            return False
    return True


# Testing with the functions from the prompt
def example_aggregate(H, messages):
    """Combine incoming messages using set union"""
    if not messages:
        return set()
    return set.union(*messages)


def example_update(U, vertex_state, aggregated_msg):
    """Update vertex state with aggregated message"""
    vertex_state["predecessors"] = aggregated_msg
    return vertex_state


def example_propagate(G, vertex_state, aggregated_msg, edge_property):
    """Propagate to successors - in this case, returns the vertex ID itself as the message"""
    vertex_id = vertex_state["id"]
    # Get all successors from the graph
    successors = list(G.successors(vertex_id))
    # Create messages for each successor
    return [(succ, {vertex_id}) for succ in successors]


# Run the algorithm
if __name__ == "__main__":
    G = create_example_graph()

    final_states = batch_vertex_centric_solver(
        G,
        example_aggregate,
        example_update,
        example_propagate,
        max_iterations=10,
        convergence_threshold=check_convergence,
    )

    print("\nFinal vertex states:")
    for node, state in final_states.items():
        print(f"Node {node}: {state}")
