from solver import BatchVertexCentricSolver
import ast
import networkx as nx
import sys


class ASTControlFlowGraph:
    """
    Class to convert a Python AST into a NetworkX graph, then use the vertex-centric
    solver to add control flow edges to the AST.
    """

    def __init__(self, source_code: str):
        """
        Initialize with Python source code.

        Args:
            source_code (str): Python source code to analyze
        """
        # Parse the source code into an AST
        self.ast_tree = ast.parse(source_code)

        # Convert AST to NetworkX graph with only AST edges
        self.graph = self._ast_to_networkx()

        # Node types that affect control flow
        self.control_flow_nodes = {
            ast.If,
            ast.For,
            ast.While,
            ast.Try,
            ast.With,
            ast.FunctionDef,
            ast.Return,
            ast.Break,
            ast.Continue,
        }

    def _ast_to_networkx(self) -> nx.DiGraph:
        """
        Convert AST to NetworkX directed graph with only AST structure edges.

        Returns:
            nx.DiGraph: NetworkX directed graph representing the AST
        """
        graph = nx.DiGraph()

        # Assign unique IDs to AST nodes and add them to the graph
        node_counter = [0]  # Use list to allow modification in inner function
        node_to_id = {}

        def add_node(node):
            node_id = node_counter[0]
            node_counter[0] += 1
            node_to_id[node] = node_id

            # Store node type and other relevant attributes
            graph.add_node(
                node_id,
                data={
                    "ast_node": node,
                    "type": type(node).__name__,
                    "lineno": getattr(node, "lineno", -1),
                    "end_lineno": getattr(node, "end_lineno", -1),
                    "next_nodes": set(),  # Will store CFG successor nodes
                    "processed": False,  # Used during CFG construction
                },
            )
            return node_id

        # Walk the AST and create nodes and edges
        def build_graph(node, parent_id=None):
            if node is None:
                return None

            node_id = add_node(node)

            # If this node has a parent, add edge from parent to this node
            if parent_id is not None:
                graph.add_edge(parent_id, node_id, type="ast")

            # Process child nodes based on node type
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    # Fields like body, orelse, etc.
                    for item in value:
                        if isinstance(item, ast.AST):
                            build_graph(item, node_id)
                elif isinstance(value, ast.AST):
                    # Single child node
                    build_graph(value, node_id)

            return node_id

        # Start building graph from the root
        build_graph(self.ast_tree)
        return graph

    def build_control_flow_graph(self):
        """
        Use the BatchVertexCentricSolver to add control flow edges to the AST graph.

        Returns:
            nx.DiGraph: The modified graph with added control flow edges
        """
        # Define algorithm-specific functions for the solver

        def init_cfg(graph):
            """Initialize analysis state for each node"""
            for node_id in graph.nodes():
                node_data = graph.nodes[node_id]["data"]
                # Each node starts unprocessed with no outgoing control flow
                node_data["processed"] = False
                node_data["next_nodes"] = set()

        def aggregate_cfg_messages(messages):
            """Aggregate potential next nodes from predecessors"""
            all_next = set()
            for next_set in messages:
                all_next.update(next_set)
            return all_next

        def update_cfg(graph, node_id, aggregated_msg):
            """Update node's next_nodes and add CFG edges to the graph"""
            node_data = graph.nodes[node_id]["data"]
            ast_node = node_data["ast_node"]

            # Mark as processed
            node_data["processed"] = True

            # Handle special node types that affect control flow
            if isinstance(ast_node, (ast.Return, ast.Break, ast.Continue)):
                node_data["next_nodes"] = set()
            else:
                for succ in aggregated_msg:
                    if (
                        not graph.has_edge(node_id, succ)
                        or graph.edges[node_id, succ]["type"] != "cfg"
                    ):
                        graph.add_edge(
                            node_id,
                            succ,
                            type="cfg",
                            cfg=str(ast_node.__class__.__name__),
                        )
                node_data["next_nodes"].update(aggregated_msg)

        def propagate_cfg(graph, node_id, aggregated_msg, edge_prop):
            """Determine control flow successors for a node"""
            node_data = graph.nodes[node_id]["data"]
            ast_node = node_data["ast_node"]
            node_type = node_data["type"]

            # Get all AST children of this node
            children = []
            for _, child_id in graph.out_edges(node_id):
                if graph.edges[node_id, child_id]["type"] == "ast":
                    children.append(child_id)

            # Find first executable child (if any)
            first_child = None
            for child in sorted(children):
                child_type = graph.nodes[child]["data"]["type"]
                if child_type not in ("Load", "Store"):  # Skip non-executable nodes
                    first_child = child
                    break

            # Default: the next node is the first child, if any
            next_nodes = {first_child} if first_child else set()

            # Special handling based on node type
            if isinstance(ast_node, ast.If):
                # If statement: Need to identify body and orelse
                body_nodes = []
                else_nodes = []

                for idx, field_name in enumerate(["body", "orelse"]):
                    if hasattr(ast_node, field_name):
                        field = getattr(ast_node, field_name)
                        if field and isinstance(field, list):
                            # Find the first node in this branch
                            for child in children:
                                child_node = graph.nodes[child]["data"]["ast_node"]
                                if child_node in field:
                                    if idx == 0:  # body
                                        body_nodes.append(child)
                                    else:  # orelse
                                        else_nodes.append(child)

                # First node in each branch is a successor
                if body_nodes:
                    next_nodes.add(min(body_nodes))
                if else_nodes:
                    next_nodes.add(min(else_nodes))

                # Both branches eventually lead to the node after the If
                # This will be added by the parent node

            elif isinstance(ast_node, (ast.For, ast.While)):
                # Loop: body and then back to the loop condition
                body_nodes = []

                if hasattr(ast_node, "body"):
                    for child in children:
                        child_node = graph.nodes[child]["data"]["ast_node"]
                        if child_node in ast_node.body:
                            body_nodes.append(child)

                # First node in body is a successor
                if body_nodes:
                    next_nodes.add(min(body_nodes))

                # The loop itself is a successor of the last node in its body
                # (handled by the parent node)

            elif isinstance(ast_node, ast.Return):
                # Return has no successors within the function
                next_nodes = set()

            elif isinstance(ast_node, ast.Break):
                # Break jumps to after the containing loop
                # Find containing loop and its parent
                loop_parent = None
                current = node_id

                # Traverse up to find containing loop
                while current is not None:
                    parent_edges = list(graph.in_edges(current))
                    if not parent_edges:
                        break

                    parent_id = parent_edges[0][0]
                    parent_node = graph.nodes[parent_id]["data"]["ast_node"]

                    if isinstance(parent_node, (ast.For, ast.While)):
                        loop_parent = parent_id
                        break

                    current = parent_id

                # Find the node after the loop (if any)
                if loop_parent is not None:
                    grandparent_edges = list(graph.in_edges(loop_parent))
                    if grandparent_edges:
                        grandparent_id = grandparent_edges[0][0]
                        children_of_grandparent = sorted(
                            [
                                child
                                for _, child in graph.out_edges(grandparent_id)
                                if graph.edges[grandparent_id, child]["type"] == "ast"
                            ]
                        )

                        # Find the node after the loop
                        for i, child in enumerate(children_of_grandparent):
                            if (
                                child == loop_parent
                                and i < len(children_of_grandparent) - 1
                            ):
                                next_nodes.add(children_of_grandparent[i + 1])
                                break

            elif isinstance(ast_node, ast.Continue):
                # Continue jumps back to the loop condition
                current = node_id

                # Traverse up to find containing loop
                while current is not None:
                    parent_edges = list(graph.in_edges(current))
                    if not parent_edges:
                        break

                    parent_id = parent_edges[0][0]
                    parent_node = graph.nodes[parent_id]["data"]["ast_node"]

                    if isinstance(parent_node, (ast.For, ast.While)):
                        next_nodes.add(parent_id)  # Loop condition is a successor
                        break

                    current = parent_id

            # If node has no special handling and no children,
            # look for siblings or parent's siblings
            if not next_nodes:
                # Get parent
                parent_edges = list(graph.in_edges(node_id))
                if parent_edges:
                    parent_id = parent_edges[0][0]
                    parent_data = graph.nodes[parent_id]["data"]

                    # Get all parent's children (our siblings)
                    siblings = sorted(
                        [
                            child
                            for _, child in graph.out_edges(parent_id)
                            if graph.edges[parent_id, child]["type"] == "ast"
                        ]
                    )

                    # Find the next sibling after this node
                    for i, sibling in enumerate(siblings):
                        if sibling == node_id and i < len(siblings) - 1:
                            next_nodes.add(siblings[i + 1])
                            break

                    # If no next sibling, propagate to parent
                    if not next_nodes:
                        next_nodes = parent_data.get("next_nodes", set())

            # Return next node set to propagate
            return [(succ, next_nodes) for succ in next_nodes if succ is not None]

        def get_vertex_ids(graph):
            """Get all vertex IDs from the graph"""
            return list(graph.nodes())

        def get_vertex_state(graph, vertex_id):
            """Extract vertex state from the graph"""
            return graph.nodes[vertex_id]["data"]

        def check_cfg_convergence(prev_graph, curr_graph):
            """Check if CFG construction has converged"""
            for node_id in prev_graph.nodes():
                if node_id not in curr_graph.nodes():
                    continue

                prev_next = prev_graph.nodes[node_id]["data"].get("next_nodes", set())
                curr_next = curr_graph.nodes[node_id]["data"].get("next_nodes", set())

                if prev_next != curr_next:
                    return False
            return True

        # Create and run the solver
        solver = BatchVertexCentricSolver(
            graph=self.graph,
            init_func=init_cfg,
            aggregate_func=aggregate_cfg_messages,
            update_func=update_cfg,
            propagate_func=propagate_cfg,
            get_vertex_ids_func=get_vertex_ids,
            get_vertex_state_func=get_vertex_state,
            edge_property_func=lambda g, s, d: None,
            max_iterations=50,
            convergence_func=check_cfg_convergence,
            debug=False,
        )

        # Run the algorithm to build the CFG
        final_graph, _ = solver.run()
        self.graph = final_graph
        return final_graph

    def visualize(self):
        """
        Create a visualization of the graph with AST and CFG edges in Graphviz DOT format.

        Returns:
            str: DOT format representation of the graph that can be rendered visually
        """
        # Start DOT format string
        dot_graph = "digraph ProgramGraph {\n"
        dot_graph += "    rankdir=TB;\n"
        dot_graph += "    node [shape=box, style=filled, fontname=Arial];\n\n"

        # Define node styles
        dot_graph += "    // Node definitions\n"
        for node_id in sorted(self.graph.nodes()):
            node_data = self.graph.nodes[node_id]["data"]
            ast_node = node_data["ast_node"]
            node_type = node_data.get("type", "unknown")

            # Create node label with detailed information
            if isinstance(ast_node, ast.AST):
                if hasattr(ast_node, "lineno"):
                    node_label = f"{type(ast_node).__name__} (line {ast_node.lineno})"
                else:
                    node_label = type(ast_node).__name__

                # Add more details for specific node types
                if isinstance(ast_node, ast.Name):
                    node_label += f"\\n{ast_node.id}"
                elif isinstance(ast_node, ast.Constant):
                    node_label += f"\\n{ast_node.value}"
                elif isinstance(ast_node, ast.FunctionDef):
                    node_label += f"\\n{ast_node.name}"
            else:
                node_label = str(ast_node)

            # Escape special characters in the label
            node_label = node_label.replace('"', '\\"')

            # Set node color based on type
            if node_type == "entry":
                color = "lightblue"
            elif node_type == "exit":
                color = "lightgreen"
            elif isinstance(ast_node, ast.FunctionDef):
                color = "gold"
            elif isinstance(ast_node, (ast.If, ast.For, ast.While)):
                color = "lightsalmon"
            else:
                color = "white"

            dot_graph += f'    {node_id} [label="{node_label}", fillcolor="{color}"];\n'

        # Add edges with different styles for AST and CFG
        dot_graph += "\n    // AST edges (solid black)\n"
        for u, v in sorted(self.graph.edges()):
            if self.graph.edges[u, v]["type"] == "ast":
                dot_graph += (
                    f'    {u} -> {v} [color="black", style="solid", label="AST"];\n'
                )

        dot_graph += "\n    // CFG edges (dashed blue)\n"
        for u, v in sorted(self.graph.edges()):
            if self.graph.edges[u, v]["type"] == "cfg":
                u_type = self.graph.nodes[u]["data"]["type"]
                v_type = self.graph.nodes[v]["data"]["type"]
                cfg_type = self.graph.edges[u, v]["cfg"]
                label = f"CFG ({cfg_type}): {u_type} -> {v_type}"
                dot_graph += (
                    f'    {u} -> {v} [color="blue", style="dashed", label="{label}"];\n'
                )

        # Close the graph
        dot_graph += "}\n"

        return dot_graph


# Example usage
def test_ast_cfg():
    # Sample Python code to analyze
    sample_code = """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
    """

    sample_code2 = """
def complex_example(x):
    total = 0
    for i in range(x):
        if i % 2 == 0:
            total += i
        else:
            total -= i
            continue
        print("Even number:", i)
    
    while total > 0:
        total -= 1
        if total < 10:
            break
    
    return total
"""

    # Create AST-CFG converter
    ast_cfg = ASTControlFlowGraph(sample_code)

    # Build the control flow graph
    ast_cfg.build_control_flow_graph()

    # Print visualization
    print(ast_cfg.visualize())


if __name__ == "__main__":
    test_ast_cfg()
