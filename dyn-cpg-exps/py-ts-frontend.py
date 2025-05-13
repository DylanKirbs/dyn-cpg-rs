from tree_sitter import Language, Parser, Node
import tree_sitter_python
from cpg import *
import html
from utils import attach_basic_colour_formatter

DEBUG_PY_TS_FRONTEND = True

PY_LANG = Language(tree_sitter_python.language())

_TRANSLATORS = {}

_NODE_KINDS = {
    "function_definition": NodeKind.FUNCTION,
    "block": NodeKind.BLOCK,
    "expression_statement": NodeKind.EXPRESSION,
    "if_statement": NodeKind.STATEMENT,
    "assignment": NodeKind.STATEMENT,
    "call": NodeKind.STATEMENT,
    "binary_operator": NodeKind.EXPRESSION,
}


def register_translator(name):
    def decorator(func):
        _TRANSLATORS[name] = func
        return func

    return decorator


# @register_translator("function_definition")
# def _tr_func_def(cpg, node, parent, order):
#     cpg_node = CPGNode(
#         kind=NodeKind.FUNCTION,
#         properties={
#             NodePropertyKey.ORDER: order,
#             NodePropertyKey.TYPE: html.escape(node.type),
#             NodePropertyKey.CODE: (
#                 html.escape(node.text.decode().strip())
#                 if node.text and node.child_count == 0
#                 else ""
#             ),
#         },
#         id=node.id,
#     )
#     cpg.addChild(parent, cpg_node)
#     for idx, child in enumerate(node.children):
#         translate(cpg, child, cpg_node, idx)


def _translate_fallback(cpg: CPG, node: Node, parent: CPGNode, order: int):

    kind = _NODE_KINDS.get(node.type, NodeKind.LANG_IMPLEMENTATION)

    cpg_node = CPGNode(
        kind=kind,
        properties={
            NodePropertyKey.ORDER: order,
            NodePropertyKey.TYPE: html.escape(node.type),
            NodePropertyKey.CODE: (
                html.escape(node.text.decode().strip())
                if node.text and node.child_count == 0
                else ""
            ),
        },
        id=node.id,
    )
    cpg_node.listeners.update(mk_dbg_lstn()) if DEBUG_PY_TS_FRONTEND else None
    cpg.addChild(parent, cpg_node)

    for idx, child in enumerate(node.children):
        translate(cpg, child, cpg_node, idx)


def translate(cpg, node, parent, order):
    _TRANSLATORS.get(node.type, _translate_fallback)(cpg, node, parent, order)


def py_to_cpg(code: str, file: str) -> CPG:

    parser = Parser(PY_LANG)
    tree = parser.parse(code.encode())

    root = CPGNode(
        kind=NodeKind.TRANSLATION_UNIT,
        properties={
            NodePropertyKey.FILE: file,
        },
        id=tree.root_node.id,
    )
    cpg = CPG(root)

    for idx, child in enumerate(tree.root_node.children):
        translate(cpg, child, root, idx)

    return cpg


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    attach_basic_colour_formatter()

    import os

    sample = """
def foo():
    x = source()
    if x > MAX:
        y = 2 * x
        sink(y)
        """

    cpg = py_to_cpg(sample, "sample.py")

    with open("cpg.dot", "w") as f:
        f.write(cpg.to_dot())

    os.system("dot -Tpdf -O cpg.dot")
