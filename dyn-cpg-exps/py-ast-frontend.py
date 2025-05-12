from cpg import *
import logging
import ast


def py_to_cpg(code: str, filename: str) -> CPG:
    """
    Converts Python code to a CPG (Code Property Graph) using the Python AST.

    Args:
        code (str): The Python code to convert.
        filename (str): The name of the file containing the code.

    Returns:
        CPG: The generated CPG.
    """

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
            setattr(node, "end_lineno", len(code.splitlines()))
        elif isinstance(node, ast.FunctionDef):
            kind = NodeKind.FUNCTION
            properties[NodePropertyKey.TYPE] = "function"
            properties[NodePropertyKey.CODE] = [node.name]
        elif isinstance(
            node,
            (
                ast.Assign,
                ast.Return,
                ast.ImportFrom,
                ast.If,
                ast.For,
                ast.While,
                ast.Compare,
                ast.arguments,
            ),
        ):
            kind = NodeKind.STATEMENT
        elif isinstance(
            node,
            (
                ast.Expr,
                ast.Expression,
                ast.Call,
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
            properties={NodePropertyKey.FILE: filename},
            listeners={
                NodePropertyKey._ALL: {
                    "debug": lambda curr_node, old_value: logging.debug(
                        "Node %s property changed: %s -> %s",
                        curr_node,
                        old_value,
                        curr_node.properties,
                    )
                }
            },
        )
    )
    module = ast.parse(code)
    for idx, node in enumerate(ast.iter_child_nodes(module)):
        convert_ast(cpg, node, parent=cpg.root, order=idx)

    return cpg


def main():

    import os

    sample = """
def foo():
    x = source()
    if x > MAX:
        y = 2 * x
        sink(y)
"""

    cpg = py_to_cpg(sample, "sample.py")
    print(cpg)

    # Print the DOT format of the CPG
    with open("cpg.dot", "w") as f:
        f.write(cpg.to_dot())

    os.system("dot -Tpdf -O cpg.dot")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
