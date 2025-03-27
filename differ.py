from subprocess import run, PIPE
from tree_sitter import Tree, Parser
from difflib import SequenceMatcher


def change_tree(old_source: bytes, new_source: bytes, parser: Parser):
    old_tree = parser.parse(old_source)

    for tag, old_start, old_end, new_start, new_end in SequenceMatcher(
        None, old_source, new_source
    ).get_opcodes():
        print(tag, old_source[old_start:old_end], new_source[new_start:new_end])

        if tag == "equal" or old_start != new_start:
            continue

        old_tree.edit(
            start_byte=old_start,
            old_end_byte=old_end,
            new_end_byte=new_end,
            start_point=(0, 0),
            old_end_point=(0, 0),
            new_end_point=(0, 0),
        )
        print(old_start, old_end, new_start, new_end)

    new_tree = parser.parse(new_source, old_tree)
    return old_tree, new_tree


if __name__ == "__main__":
    import tree_sitter_python as tspy
    from tree_sitter import Language

    py_parser = Parser(Language(tspy.language()))

    old_source = b"def foo():\n    if x > y:\n        print('hello')\n"
    new_source = b"def foo():\n    if x < y:\n        print('Hello')\n"

    old_tree, new_tree = change_tree(old_source, new_source, py_parser)
    print(old_tree.changed_ranges(new_tree))
