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
from contextlib import contextmanager

# Helpers


def log_nyi(func):
    """
    Decorator to log that a function is not yet implemented.
    """

    def wrapper(*args, **kwargs):
        logging.critical(f"{func.__name__} NOT YET IMPLEMENTED")
        return func(*args, **kwargs)

    return wrapper


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

    _ALL = "*"
    """Wildcard for all properties, must not be set, only to be used by listeners"""

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
    str, int, float, bool, List[str], List[int], List[float], List[bool], None
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
        NodePropertyKey, Dict[str, Callable[["CPGNode", PropertyValue], None]]
    ] = field(default_factory=dict)
    """The listeners for this node, if any. In the format {property: {name:listener}}, where listener gets called with the node and the old value of the property. The name is used to uniquely identify the listener so that it can be managed."""

    _listeners_suspended: bool = field(default=False, init=False, repr=False)
    _pending_notifications: List[Tuple[NodePropertyKey, PropertyValue]] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def id(self) -> int:
        """
        The unique identifier of the node.

        Returns:
            int: The unique identifier of the node.
        """
        return id(self)

    def children(self, cpg) -> List["CPGNode"]:
        """
        Get the children of this node in the CPG.

        Args:
            cpg (CPG): The CPG to get the children from.

        Returns:
            List[CPGNode]: The children of this node in the CPG.
        """

        children = []
        for s, t in cpg.edges:
            if s == self.id and any(
                [e.kind == EdgeKind.SYNTAX for e in cpg.edges[(s, t)]]
            ):
                children.append(cpg.nodes[t])

        return children

    @contextmanager
    def suspend_listeners(self):
        """
        Context manager to temporarily suspend all listeners for this node.
        """
        self._listeners_suspended = True
        try:
            yield
        finally:
            self._listeners_suspended = False
            # Re-fire all listeners for changed keys
            for key, old_value in self._pending_notifications:
                for _, listener in self.listeners.get(NodePropertyKey._ALL, {}).items():
                    listener(self, old_value)
                for _, listener in self.listeners.get(key, {}).items():
                    listener(self, old_value)
            self._pending_notifications.clear()

    def _update_property_and_notify(
        self, key: NodePropertyKey, value: PropertyValue
    ) -> None:
        old_value = self.properties.get(key)
        self.properties[key] = value

        if self._listeners_suspended:
            self._pending_notifications.append((key, old_value))
            return

        for _, listener in self.listeners.get(NodePropertyKey._ALL, {}).items():
            listener(self, old_value)

        for _, listener in self.listeners.get(key, {}).items():
            listener(self, old_value)

    def _add_or_update_listener(
        self, key: NodePropertyKey, name: str, listener: Callable
    ):
        """
        Update or add a listener for a property of the node.

        Args:
            key (NodePropertyKey): The key of the property to listen to.
            name (str): The name of the listener.
            listener (Callable): The listener function to call when the property changes.
        """

        if key not in self.listeners:
            self.listeners[key] = {}

        self.listeners[key][name] = listener

    def _remove_listener(self, key: NodePropertyKey, name: str):
        """
        Remove a listener for a property of the node.

        Args:
            key (NodePropertyKey): The key of the property to remove the listener from.
            name (str): The name of the listener to remove.
        """

        if key in self.listeners and name in self.listeners[key]:
            del self.listeners[key][name]

    def subscribe_order_to(self: "CPGNode", left: "CPGNode"):
        """The right (self) node subscribes to be notified when the left node changes order"""

        right = self
        if right.id == left.id:
            logging.warning(
                "Aborting subscription setup, node must not be subscribed to itself: %s",
                right.id,
            )
            return

        def update_node_order(updated_left, old_left_order):
            if updated_left.id == right.id:
                logging.debug(
                    "Aborting, node order must not be subscribed to itself: %s %s",
                    updated_left.id,
                    updated_left.listeners,
                )
                return
            # Update the order of the right node and notify its listeners
            right._update_property_and_notify(
                NodePropertyKey.ORDER,
                right.properties.get(NodePropertyKey.ORDER, 0)
                + updated_left.properties.get(NodePropertyKey.ORDER, 0)
                - (old_left_order or 0),
            )

        left._remove_listener(NodePropertyKey.ORDER, "sibling_order")
        left._add_or_update_listener(
            NodePropertyKey.ORDER,
            "sibling_order",
            update_node_order,
        )

    @log_nyi
    def propagate_insert(self, cpg: "CPG", parent: "CPGNode") -> None:
        """
        Called after the node is inserted into the CPG.

        Args:
            cpg (CPG): The CPG node.
            parent (CPGNode): The parent node.
        """

        logging.debug("Propagating node insert %s under parent %s", self.id, parent.id)

        if parent.id not in cpg.nodes:
            raise ValueError(f"Parent node {parent.id} not found in CPG")

        if self.id not in cpg.nodes:
            raise ValueError(f"Node {self.id} not found in CPG")

        # Subscribe myself to my predecessors order, and my successor to my order
        siblings = sorted(
            filter(lambda x: x is not self and x.id != self.id, parent.children(cpg)),
            key=lambda x: x.properties.get(NodePropertyKey.ORDER, -1),  # type: ignore
        )
        left_sibling: Optional["CPGNode"] = None
        right_sibling: Optional["CPGNode"] = None
        for sibling in siblings:

            if not right_sibling and sibling.properties.get(
                NodePropertyKey.ORDER, 0
            ) == self.properties.get(NodePropertyKey.ORDER, 0):
                right_sibling = sibling
            elif (
                not left_sibling
                and sibling.properties.get(NodePropertyKey.ORDER, 0)
                == self.properties.get(NodePropertyKey.ORDER, 0) - 1  # type: ignore
            ):
                left_sibling = sibling

            if left_sibling and right_sibling:
                break

        if left_sibling:
            self.subscribe_order_to(left_sibling)

        if right_sibling:
            right_sibling.subscribe_order_to(self)

            logging.debug(
                "Shifting right sibling [%s] of [%s] and notifying",
                right_sibling.id,
                self.id,
            )
            right_sibling._update_property_and_notify(
                NodePropertyKey.ORDER,
                self.properties.get(NodePropertyKey.ORDER, 0) + 1,  # type: ignore
            )

        # Notify other nodes about the insertion
        self._update_property_and_notify(NodePropertyKey.PARENT_ID, parent.id)

        # TODO: The rest :)

    @log_nyi
    def propagate_update(
        self, cpg: "CPG", what: Literal["addChild", "addEdge"], **kwargs
    ) -> None:
        """
        Called when the node is updated in the CPG.

        Args:
            cpg (CPG): The CPG to update the node in.
            what (str): The type of update to perform.
            kwargs: Additional arguments for the update.
        """

        kwarg_requirements = {
            "addChild": {"child": CPGNode},
            "addEdge": {"edge": CPGEdge},
        }

        logging.debug("Propagating node update %s", self.id)

        if what not in ["addChild", "addEdge"]:
            raise ValueError(f"Invalid update type: {what}")

        for name, type in kwarg_requirements[what].items():
            if name not in kwargs:
                raise ValueError(f"Missing required argument: {name}")
            if not isinstance(kwargs[name], type):
                raise ValueError(f"Invalid type for argument {name}: {type}")

        if self.id not in cpg.nodes:
            raise ValueError(f"Node {self.id} not found in CPG")

        # TODO: Notify other nodes about the update
        ...

    @log_nyi
    def propagate_delete(self, cpg: "CPG") -> None:
        """
        Called when the node is deleted from the CPG.

        Args:
            cpg (CPG): The CPG to delete the node from.
        """

        logging.debug("Propagating node delete %s", self.id)

        if self.id not in cpg.nodes:
            raise ValueError(f"Node {self.id} not found in CPG")

        # TODO: Notify other nodes about the deletion
        ...

    def to_dot(self) -> str:
        """
        Convert the node to a DOT format string.
        This can be used to visualise the node using Graphviz.

        Returns:
            str: The DOT format string representing the node.
        """

        col = "black"
        bg = "white"
        if self.kind == NodeKind.TRANSLATION_UNIT:
            bg = "orange"
        elif self.kind == NodeKind.LANG_IMPLEMENTATION:
            col = "grey"

        dot = f'  {self.id} [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        dot += f"<TR><TD>{self.kind.name}</TD></TR>"
        dot += f"<TR><TD>{self.properties}</TD></TR>"
        dot += f"</TABLE>>, color={col}, style=filled, fillcolor={bg}]\n"
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
        dot += "  rankdir=LR\n"
        dot += "  ranksep=0.5\n"
        dot += "  nodesep=0.5\n"

        for n in sorted(self.nodes.values(), key=lambda x: x.properties.get(NodePropertyKey.ORDER, -1)):  # type: ignore
            dot += n.to_dot()

        for es in self.edges.values():
            for e in es:
                dot += e.to_dot()

        # my patent pending rankifyer XD
        # starting from the root, combine ranks by parent
        # i.e. root : a, b and a : c and b : d then c and d are in the same rank
        ranks = {self.root.id: 0}
        queue = [self.root.id]
        while queue:
            node_id = queue.pop(0)
            for k, v in self.edges.items():
                if not any(
                    [e.kind == EdgeKind.SYNTAX for e in v if e.source.id == node_id]
                ):
                    continue
                if k[0] == node_id:
                    # we have a child, so it's rank is ours + 1
                    child_id = k[1]
                    if child_id not in ranks:
                        ranks[child_id] = ranks[node_id] + 1
                        queue.append(child_id)

        rev_ranks = {}
        for k, v in ranks.items():
            if v not in rev_ranks:
                rev_ranks[v] = []
            rev_ranks[v].append(k)

        for same_rank in rev_ranks.values():
            if len(same_rank) > 1:
                dot += f"  {{rank=same; {' '.join(map(str, same_rank))}}}\n"

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

        self.nodes[child.id] = child
        e = CPGEdge(source=parent, target=child, kind=EdgeKind.SYNTAX)
        self._ins_edge(e)

        parent.propagate_update(self, "addChild", child=child)
        child.propagate_insert(self, parent)

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

        source.propagate_update(self, "addEdge", edge=e)
        target.propagate_update(self, "addEdge", edge=e)


def main():
    """
    Main function to demonstrate the CPG functionality.
    """

    import os

    dbg_lstn = {
        NodePropertyKey._ALL: {
            "debug": lambda curr_node, old_value: logging.debug(
                "LISTENER LOG EVENT: Node %s property changed: %s -> %s",
                curr_node.id,
                old_value,
                curr_node.properties,
            )
        }
    }

    # Create a simple CPG
    root = CPGNode(kind=NodeKind.TRANSLATION_UNIT, listeners=dbg_lstn)
    cpg = CPG(ast_root=root)

    # Create some nodes
    func_node = CPGNode(
        kind=NodeKind.FUNCTION,
        properties={NodePropertyKey.ORDER: 0},
        listeners=dbg_lstn,
    )
    stmt1_node = CPGNode(
        kind=NodeKind.STATEMENT,
        properties={NodePropertyKey.ORDER: 0},
        listeners=dbg_lstn,
    )
    stmt2_node = CPGNode(
        kind=NodeKind.STATEMENT,
        properties={NodePropertyKey.ORDER: 1},
        listeners=dbg_lstn,
    )

    # Add nodes to the CPG
    cpg.addChild(root, func_node)
    cpg.addChild(func_node, stmt1_node)
    cpg.addChild(func_node, stmt2_node)

    # Print the CPG in DOT format
    with open("cpg.dot", "w") as f:
        f.write(cpg.to_dot())

    os.system("dot -Tpdf -O cpg.dot")
    input("Press Enter to continue...")

    # Now, the magic! An incremental update to the CPG, add an expression in between the statements
    expr_node = CPGNode(
        kind=NodeKind.EXPRESSION,
        properties={NodePropertyKey.ORDER: 1},
        listeners=dbg_lstn,
    )
    cpg.addChild(func_node, expr_node)

    # Print the CPG in DOT format
    with open("cpg.dot", "w") as f:
        f.write(cpg.to_dot())

    os.system("dot -Tpdf -O cpg.dot")
    input("Press Enter to continue...")

    expr_node = CPGNode(
        kind=NodeKind.BLOCK,
        properties={NodePropertyKey.ORDER: 1},
        listeners=dbg_lstn,
    )
    cpg.addChild(func_node, expr_node)

    # Print the CPG in DOT format
    with open("cpg.dot", "w") as f:
        f.write(cpg.to_dot())

    os.system("dot -Tpdf -O cpg.dot")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
