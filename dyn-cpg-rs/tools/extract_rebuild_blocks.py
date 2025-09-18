#!/usr/bin/env python3
from argparse import ArgumentParser
import sys
from pathlib import Path
import logging
import re


def main(argv=None) -> int:
    parser = ArgumentParser(
        description="Extract REBUILD mapping blocks that contain a token from a rebuild log"
    )
    parser.add_argument(
        "token", type=str, help="Token to search for (or regex if --regex)"
    )
    parser.add_argument("log", type=Path, help="Path to the rebuild log file")
    parser.add_argument(
        "--delimiter",
        type=str,
        default="\n=== REBUILD mapping:",
        help="Block delimiter used to split the log",
    )
    parser.add_argument(
        "--max-blocks", type=int, default=50, help="Maximum number of blocks to print"
    )
    parser.add_argument(
        "--regex", action="store_true", help="Treat token as a regular expression"
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Case-insensitive matching for regex or substring",
    )
    parser.add_argument(
        "--exit-on-match",
        action="store_true",
        help="Return exit code 1 when matches are found (useful for CI)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )

    token = args.token
    log_path = args.log

    if not log_path.exists():
        logging.error("Log file not found: %s", log_path)
        return 2

    if token == "":
        logging.error("Empty token not allowed")
        return 2

    try:
        text = log_path.read_text(errors="replace")
    except Exception as e:
        logging.error("Could not read log file %s: %s", log_path, e)
        return 2

    delimiter = args.delimiter
    blocks = text.split(delimiter)

    matches = []
    if args.regex:
        flags = re.MULTILINE
        if args.ignore_case:
            flags |= re.IGNORECASE
        try:
            pat = re.compile(token, flags)
        except re.error as e:
            logging.error("Invalid regex %s: %s", token, e)
            return 2
        for b in blocks:
            if pat.search(b):
                matches.append(delimiter + b)
    else:
        if args.ignore_case:
            t_lower = token.lower()
            for b in blocks:
                if t_lower in b.lower():
                    matches.append(delimiter + b)
        else:
            for b in blocks:
                if token in b:
                    matches.append(delimiter + b)

    if not matches:
        logging.info("No blocks found containing token %s", token)
        return 0

    total = len(matches)
    logging.info("Found %d blocks containing %s", total, token)

    for i, m in enumerate(matches[: args.max_blocks], 1):
        print(f"--- MATCH {i}/{total} ---")
        print(m.strip())
        print()

    if total > args.max_blocks:
        logging.info(
            "(truncated output: showed %d of %d matches)", args.max_blocks, total
        )

    if args.exit_on_match:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
