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
from typing import Dict, List, Optional, Tuple, Union, Callable
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

    start_byte: int
    """The start byte of the node in the source code"""

    end_byte: int
    """The end byte of the node in the source code"""

    order: int
    """The order of this node in the parent node's children, if any"""

    parent: Optional["CPGNode"] = None
    """The parent node of this node, if any"""

    children: List["CPGNode"] = field(default_factory=list)
    """The children of this node, if any"""

    properties: Dict[NodePropertyKey, PropertyValue] = field(default_factory=dict)
    """The properties of this node, if any"""

    in_edges: List["CPGEdge"] = field(default_factory=list)
    """The incoming edges of this node, if any"""

    out_edges: List["CPGEdge"] = field(default_factory=list)
    """The outgoing edges of this node, if any"""

    def insert(self, *args, **kwargs):
        """
        Insert a new node into the CPG.

        Handles the creation of the new node and performs the relevant steps to update the CPG.
        This includes updating the parent node's children, and linking the new node to its parent.
        As well as updating the CF and DF edges to and from this node.
        """

        logging.debug("Inserting node: %s", self)

        # TODO: Implement the logic for inserting this node
        logging.error("Node insertion not implemented")

    def update(self, *args, **kwargs):
        """
        Update node event.

        Handles the update of the node and performs the relevant steps to update the CPG.
        """

        logging.debug("Updating node: %s", self)

        # TODO: Implement the logic for updating this node
        logging.error("Node update not implemented")

    def delete(self, *args, **kwargs) -> None:
        """
        Delete node event.

        Handles the removal of the node and performs the relevant steps to update the CPG.
        This includes removing the node from its parent, and re-linking all edges to and from this node.
        """

        logging.debug("Deleting node: %s", self)

        # TODO: Implement the logic for deleting this node
        logging.error("Node deletion not implemented")

        ...

        del self

    def to_dot(self, keep_lang_impl=False) -> str:
        """
        Convert the node to a DOT format string.
        This can be used to visualise the node using Graphviz.

        Returns:
            str: The DOT format string representing the node.
        """

        if self.kind == NodeKind.LANG_IMPLEMENTATION and not keep_lang_impl:
            return ""

        dot = f'  {id(self)} [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        dot += f"<TR><TD>{self.kind.name}</TD></TR>"
        dot += f"<TR><TD>{self.start_byte}:{self.end_byte}</TD></TR>"
        dot += f"<TR><TD>{self.order}</TD></TR>"
        dot += f"<TR><TD>{self.properties}</TD></TR>"
        dot += "</TABLE>>]\n"

        dot += "{ rank=same; "
        for child in self.children:
            if child.kind == NodeKind.LANG_IMPLEMENTATION and not keep_lang_impl:
                continue
            dot += f"{id(child)}; "
        dot += "}\n"
        for child in self.children:
            if child.kind == NodeKind.LANG_IMPLEMENTATION and not keep_lang_impl:
                continue
            dot += child.to_dot()

        for edge in self.out_edges:
            if edge.target.kind == NodeKind.LANG_IMPLEMENTATION and not keep_lang_impl:
                continue
            if edge.source.kind == NodeKind.LANG_IMPLEMENTATION and not keep_lang_impl:
                continue

            el = edge.kind.name
            if edge.kind == EdgeKind.CONTROL_FLOW:
                el = f"CF_{edge.properties[EdgePropertyKey.CONTROL_DEPENDENCE]}"
            dot += f'  {id(self)} -> {id(edge.target)} [label="{el}"]\n'

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

    def update_nodes(self) -> None:
        """
        Update the source and target nodes of the edge.
        This is called when the edge is created or modified.
        """
        self.source.out_edges.append(self)
        self.target.in_edges.append(self)


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
        dot += f"  {self.root.to_dot()}\n"
        dot += "}\n"

        return dot


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
    def convert_ast(
        node: ast.AST, parent: Optional[CPGNode] = None, order: int = 0
    ) -> CPGNode:
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
        cpg_node = CPGNode(
            kind=kind,
            start_byte=getattr(node, "lineno", 0),
            end_byte=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
            order=order,
            parent=parent,
            properties=properties,
        )

        # Recursively process child nodes
        for idx, child in enumerate(ast.iter_child_nodes(node)):
            child_node = convert_ast(child, parent=cpg_node, order=idx)
            cpg_node.children.append(child_node)
            e = CPGEdge(source=cpg_node, target=child_node, kind=EdgeKind.SYNTAX)
            e.update_nodes()

        return cpg_node

    # Convert the AST to a CPG
    cpg_root = convert_ast(ast.parse(sample))
    cpg = CPG(cpg_root)

    # Print the DOT format of the CPG
    with open("cpg.dot", "w") as f:
        f.write(cpg.to_dot())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
