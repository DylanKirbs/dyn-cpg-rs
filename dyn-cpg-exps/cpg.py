"""
A dynamic Code Property Graph (CPG) implementation for Python.

This module provides the facilities to create and manipulate a CPG, which is a representation of the code structure and flow.
It includes classes for nodes, edges, and the graph itself, as well as methods for interacting with nodes and edges, and generating the graph.
Additionally, an API is provided for updating the underlying Abstract Syntax Tree (AST) of an existing CPG, and propagating changes through the graph.

This CPG implementation expects the user to conform to a specific AST structure, thus any language-specific AST must be converted by a suitable language frontend.
Facilities are provided to serialise the CPG to a graph database, such as Neo4j, and to export the CPG to a JSON format for further analysis or storage.

[WARNING]:
- This is an experimental proof of concept, and is not yet optimised or stable.
- This is a simplified model of a CPG, and does not include all the features of a full CPG implementation.

@module cpg
@author: Dylan Kirby [25853805@sun.ac.za]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union, Callable
from enum import Enum, auto
import logging

# Nodes & Edges


class NodeKind(Enum):
    """
    Enum for the types of nodes in the CPG.
    """

    TRANSLATION_UNIT = auto()
    """The root node of the CPG"""

    FUNCTION = auto()
    """A function node in the CPG"""

    BLOCK = auto()
    """A block node in the CPG"""

    STATEMENT = auto()
    """A statement node in the CPG"""

    EXPRESSION = auto()
    """An expression node in the CPG"""

    LANG_IMPLEMENTATION = auto()
    """A language implementation node in the CPG
    We don't care what this is or how it works,
    it just serves to keep track of language specific features
    that could be handy for analysis"""


class EdgeKind(Enum):
    """
    Enum for the types of edges in the CPG.
    """

    SYNTAX = auto()
    """An edge representing the AST structure"""

    CONTROL_FLOW = auto()
    """An edge representing the control flow"""

    PROGRAM_DEPENDENCE = auto()
    """An edge representing the program dependence"""


# Properties


class NodePropertyKey(Enum):
    """
    Enum for the keys of properties in a node.
    """

    TYPE = "type"
    """The data type of the node (e.g. int, float, etc.)"""

    FILE = "file"
    """The file in which the node is defined (for a translation unit)"""

    START_BYTE = "start_byte"
    """The start byte offset of the node in the source code"""

    END_BYTE = "end_byte"
    """The end byte offset of the node in the source code"""

    ORDER = "order"
    """The order of the node among its siblings"""
    """The code associated with the node (e.g. the source code)"""

    CODE = "code"
    """The code associated with the node (e.g. the source code)"""

    PARENT_ID = "parent_id"
    """The parent node of this node"""

    def __repr__(self):
        return f"{self.value}"


class EdgePropertyKey(Enum):
    """
    Enum for the keys of properties in an edge.
    """

    DATA_DEPENDENCE = "data_dependence"
    """The data dependence of the edge (e.g. the identifier of the data dependency)"""

    CONTROL_DEPENDENCE = "control_dependence"
    """The control dependence of the edge (e.g. true, false, etc.)"""

    def __repr__(self):
        return f"{self.value}"


PropertyValue = Union[
    str, int, float, bool, List[str], List[int], List[float], List[bool]
]


@dataclass
class CPGNode:
    """
    A node in the CPG.
    """

    kind: NodeKind
    """The type of the node"""

    properties: Dict[NodePropertyKey, PropertyValue] = field(default_factory=dict)
    """The properties of this node, if any"""

    listeners: Dict[
        NodePropertyKey, List[Callable[["CPGNode", PropertyValue], None]]
    ] = field(default_factory=dict)
    """The listeners for this node, if any. In the format {property: [listeners]}, where listener gets called with the node and the old value of the property"""

    @property
    def id(self) -> int:
        """
        The unique identifier of the node.

        Returns:
            int: The unique identifier of the node.
        """
        return id(self)

    # TODO: determine if it is better to just have a function for each mode instead
    def update(
        self,
        mode: Literal["insert", "addChild", "addEdge", "delete"],
        cpg: "CPG",
        **kwargs,
    ) -> None:
        """
        Update the node in the CPG.

        Handles the update of the node and performs the relevant steps to update the CPG.
        This includes updating the node's properties, and re-linking all edges to and from this node.
        And notifying subscribers of the update.
        """

        logging.debug("Updating node: %s [%s]", self, mode)

        if mode == "insert":
            # We are inserting the node into the CPG and need to update our parent and siblings
            parent = kwargs.get("parent")
            if parent is None:
                raise ValueError("Parent node is required for insert mode")

            if self.properties.get(NodePropertyKey.ORDER) is None:
                raise ValueError("Order property is required for insert mode")
            order: int = self.properties[NodePropertyKey.ORDER]  # type: ignore

            if order > 0:
                # TODO: this is a hack and is not really a true representation
                # We need to find the previous sibling and add an edge to it
                siblings = [
                    n
                    for n in cpg.nodes.values()
                    if n.properties.get(NodePropertyKey.ORDER) == order - 1
                    and n.properties.get(NodePropertyKey.PARENT_ID) == parent.id
                ]
                if not siblings:
                    raise ValueError("No previous sibling found")
                prev_sibling = siblings[0]
                cpg.addEdge(prev_sibling, self, EdgeKind.CONTROL_FLOW)

            # TODO
            return

        if mode == "addChild":
            # We are adding a child node to this node
            child = kwargs.get("child")
            if child is None:
                raise ValueError("Child node is required for addChild mode")

            # TODO
            return

        if mode == "delete":
            # We are deleting the node from the CPG
            # TODO

            del self
            return

    def to_dot(self) -> str:
        """
        Convert the node to a DOT format string.
        This can be used to visualise the node using Graphviz.

        Returns:
            str: The DOT format string representing the node.
        """

        dot = f'  {self.id} [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        dot += f"<TR><TD>{self.kind.name}</TD></TR>"
        dot += f"<TR><TD>{self.properties}</TD></TR>"
        dot += "</TABLE>>]\n"

        return dot


@dataclass
class CPGEdge:
    """
    An edge in the CPG.
    """

    source: CPGNode
    """The source node of the edge"""

    target: CPGNode
    """The target node of the edge"""

    kind: EdgeKind
    """The type of the edge"""

    properties: Dict[EdgePropertyKey, PropertyValue] = field(default_factory=dict)
    """The properties of this edge, if any"""

    def to_dot(self) -> str:
        """
        Convert the edge to a DOT format string.
        This can be used to visualise the edge using Graphviz.

        Returns:
            str: The DOT format string representing the edge.
        """

        col = "black"
        if self.kind == EdgeKind.CONTROL_FLOW:
            col = "blue"
        elif self.kind == EdgeKind.PROGRAM_DEPENDENCE:
            col = "red"

        dot = f'  {self.source.id} -> {self.target.id} [label="{self.kind.name}", color={col}]\n'
        return dot


class CPG:
    """
    A Code Property Graph (CPG) implementation.
    """

    def __init__(self, ast_root: CPGNode):
        """
        Create a new CPG from an AST root node.

        Args:
            ast_root (CPGNode): The root node of the AST.
        """

        self.root = ast_root
        self.nodes: Dict[int, CPGNode] = {ast_root.id: ast_root}
        self.edges: Dict[Tuple[int, int], List[CPGEdge]] = {}

    def to_dot(self) -> str:
        """
        Convert the CPG to a DOT format string.
        This can be used to visualise the CPG using Graphviz.
        Returns:
            str: The DOT format string representing the CPG.
        """

        dot = "digraph G {\n"
        dot += "  node [shape=rectangle]\n"
        dot += "  edge [arrowhead=vee]\n"
        dot += "  rankdir=TB\n"
        dot += "  ranksep=0.5\n"
        dot += "  nodesep=0.5\n"

        for n in self.nodes.values():
            dot += n.to_dot()

        for es in self.edges.values():
            for e in es:
                dot += e.to_dot()

        dot += "}\n"

        return dot

    def _ins_edge(self, e: CPGEdge) -> None:
        k = (e.source.id, e.target.id)
        if k not in self.edges:
            self.edges[k] = []
        if e not in self.edges[k]:
            self.edges[k].append(e)

    def __repr__(self) -> str:
        """
        Return a string representation of the CPG.
        This includes the number of nodes and edges in the CPG.

        Returns:
            str: A string representation of the CPG.
        """

        num_e = sum(len(es) for es in self.edges.values())

        return f"CPG with {len(self.nodes)} nodes and {num_e} edges"

    def addChild(self, parent: CPGNode, child: CPGNode) -> None:
        """
        Add a child node to a parent node in the CPG and create a 'syntax' edge between them.

        Args:
            parent (CPGNode): The parent node.
            child (CPGNode): The child node to add.
        """

        if parent.id not in self.nodes:
            raise ValueError(f"Parent node {parent.id} not found in CPG")

        child.properties[NodePropertyKey.PARENT_ID] = parent.id

        self.nodes[child.id] = child
        e = CPGEdge(source=parent, target=child, kind=EdgeKind.SYNTAX)
        self._ins_edge(e)

        parent.update("addChild", self, child=child)
        child.update("insert", self, parent=parent)

    def addEdge(self, source: CPGNode, target: CPGNode, kind: EdgeKind) -> None:
        """
        Add an edge between two nodes in the CPG.

        Args:
            source (CPGNode): The source node.
            target (CPGNode): The target node.
            kind (EdgeKind): The type of the edge.
        """

        if source.id not in self.nodes or target.id not in self.nodes:
            raise ValueError("Source or target node not found in CPG")

        e = CPGEdge(source=source, target=target, kind=kind)
        self._ins_edge(e)

        source.update("addEdge", self, edge=e)
        target.update("addEdge", self, edge=e)


def main():

    import ast

    sample = """
from typing import List

def foo(x: int) -> List[int]:
    # A very cool function that does math
    y = x + 1
    if y > 10:
        y = 10
    else:
        y = 5
    z = y * 2
    return [y, z]
    
foo(5)
"""

    # As the "language frontend" we MUST convert the python AST to a format the CPG can use
    def convert_ast(cpg: CPG, node: ast.AST, parent: CPGNode, order: int = 0):
        """
        Recursively converts a Python AST node into a CPGNode and its children.

        Args:
            node (ast.AST): The Python AST node to convert.
            parent (Optional[CPGNode]): The parent CPGNode, if any.
            order (int): The order of the node among its siblings.

        Returns:
            CPGNode: The root CPGNode of the converted subtree.
        """
        kind = NodeKind.LANG_IMPLEMENTATION  # Default kind for generic nodes
        properties: Dict[NodePropertyKey, PropertyValue] = {
            NodePropertyKey.TYPE: node.__class__.__name__.lower()
        }

        if isinstance(node, ast.Module):
            kind = NodeKind.TRANSLATION_UNIT
            properties[NodePropertyKey.FILE] = "sample.py"
            setattr(node, "end_lineno", len(sample.splitlines()))
        elif isinstance(node, ast.FunctionDef):
            kind = NodeKind.FUNCTION
            properties[NodePropertyKey.TYPE] = "function"
            properties[NodePropertyKey.CODE] = [node.name]
        elif isinstance(node, (ast.If, ast.For, ast.While)):
            kind = NodeKind.BLOCK
        elif isinstance(
            node,
            (
                ast.BinOp,
                ast.Expr,
                ast.Expression,
                ast.Assign,
                ast.Return,
                ast.ImportFrom,
            ),
        ):
            kind = NodeKind.STATEMENT
        elif isinstance(
            node,
            (
                ast.Name,
                ast.Constant,
                ast.List,
                ast.Tuple,
                ast.Subscript,
                ast.Attribute,
                ast.Call,
                ast.UnaryOp,
                ast.BinOp,
                ast.Compare,
                ast.BoolOp,
                ast.ListComp,
            ),
        ):
            kind = NodeKind.EXPRESSION

        # Create the current CPGNode
        properties.update(
            {
                NodePropertyKey.START_BYTE: getattr(node, "lineno", 0),
                NodePropertyKey.END_BYTE: getattr(
                    node, "end_lineno", getattr(node, "lineno", 0)
                ),
                NodePropertyKey.ORDER: order,
            }
        )
        cpg_node = CPGNode(
            kind=kind,
            properties=properties,
        )
        cpg.addChild(parent, cpg_node)

        # Recursively process child nodes
        for idx, child in enumerate(ast.iter_child_nodes(node)):
            convert_ast(cpg, child, parent=cpg_node, order=idx)

    # Convert the AST to a CPG
    cpg = CPG(
        CPGNode(
            kind=NodeKind.TRANSLATION_UNIT,
            properties={NodePropertyKey.FILE: "sample.py"},
        )
    )
    module = ast.parse(sample)
    for idx, node in enumerate(ast.iter_child_nodes(module)):
        convert_ast(cpg, node, parent=cpg.root, order=idx)

    print(cpg)

    # Print the DOT format of the CPG
    with open("cpg.dot", "w") as f:
        f.write(cpg.to_dot())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
