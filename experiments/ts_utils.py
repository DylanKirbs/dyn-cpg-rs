import tree_sitter_python
import tree_sitter_c
from tree_sitter import Language, Parser


class Languages:
    python = Language(tree_sitter_python.language())
    c = Language(tree_sitter_c.language())

    supported_langs = {
        "python": python,
        "c": c,
    }


class Parsers:
    python = Parser(Languages.python)
    c = Parser(Languages.c)
