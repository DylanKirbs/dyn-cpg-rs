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

    def update(self, mode: Literal["insert", "addChild", "delete"], **kwargs) -> None:
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

    def to_dot(self, keep_lang_impl=False) -> str:
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
        self.edges: Dict[Tuple[int, int], CPGEdge] = {}

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
        dot += "  rankdir=LR\n"
        dot += "  ranksep=0.5\n"
        dot += "  nodesep=0.5\n"

        for node in self.nodes.values():
            dot += node.to_dot()

        for edge in self.edges.values():
            dot += (
                f'  {edge.source.id} -> {edge.target.id} [label="{edge.kind.name}"]\n'
            )

        dot += "}\n"

        return dot

    def addChild(self, parent: CPGNode, child: CPGNode) -> None:
        """
        Add a child node to a parent node in the CPG and create a 'syntax' edge between them.

        Args:
            parent (CPGNode): The parent node.
            child (CPGNode): The child node to add.
        """

        if parent.id not in self.nodes:
            raise ValueError(f"Parent node {parent.id} not found in CPG")

        self.nodes[child.id] = child
        self.edges[(parent.id, child.id)] = CPGEdge(
            source=parent, target=child, kind=EdgeKind.SYNTAX
        )

        parent.update("addChild", child=child)
        child.update("insert", parent=parent)


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

    # Print the DOT format of the CPG
    with open("cpg.dot", "w") as f:
        f.write(cpg.to_dot())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
