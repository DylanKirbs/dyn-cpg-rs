#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
import logging
import sys


def find_all_indexes(data: bytes, needle: bytes):
    """Yield all start indexes of needle in data (including overlapping)."""
    if len(needle) == 0:
        return
    start = 0
    while True:
        idx = data.find(needle, start)
        if idx == -1:
            break
        yield idx
        start = idx + 1


def main(argv=None) -> int:
    parser = ArgumentParser(
        description="Find raw byte occurrences and NodeId( entries in a file"
    )
    parser.add_argument("haystack", type=Path)
    parser.add_argument(
        "needle",
        type=str,
        nargs="?",
        default=None,
        help="UTF-8 string needle to search for (unless --hex is used)",
    )
    parser.add_argument(
        "--hex",
        action="store_true",
        help="Interpret needle as hex (e.g. deadbeef) and search bytes",
    )
    parser.add_argument(
        "--context", type=int, default=40, help="Bytes of surrounding context to print"
    )
    parser.add_argument(
        "--max-matches", type=int, default=20, help="Maximum matches to print"
    )
    parser.add_argument(
        "--nodeid",
        action="store_true",
        help="Also search for literal 'NodeId(' occurrences",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )

    p = args.haystack
    if not p.exists():
        logging.error("File does not exist: %s", p)
        return 2

    try:
        b = p.read_bytes()
    except Exception as e:
        logging.error("Could not read file %s: %s", p, e)
        return 2

    logging.info("file size %d", len(b))

    # prepare needle
    needle_bytes = None
    if args.needle is not None:
        if args.hex:
            try:
                needle_bytes = bytes.fromhex(args.needle)
            except ValueError:
                logging.error("Invalid hex needle: %s", args.needle)
                return 2
        else:
            needle_bytes = args.needle.encode("utf-8")

    # search for needle if provided
    if needle_bytes is not None:
        idxs = list(find_all_indexes(b, needle_bytes))
        logging.info("raw matches for %s : %d", args.needle, len(idxs))
        for i in idxs[: args.max_matches]:
            start = max(0, i - args.context)
            end = min(len(b), i + args.context)
            logging.info("--- match at %d", i)
            logging.info(repr(b[start:end]))

    # search for 'NodeId('
    if args.nodeid:
        nid = b"NodeId("
        idxs2 = list(find_all_indexes(b, nid))
        logging.info("NodeId( matches: %d", len(idxs2))
        for i in idxs2[: args.max_matches]:
            start = max(0, i - args.context)
            end = min(len(b), i + args.context + 20)
            logging.info("--- NodeId at %d", i)
            logging.info(repr(b[start:end]))

    # exit code: 0 success
    return 0


if __name__ == "__main__":
    sys.exit(main())
