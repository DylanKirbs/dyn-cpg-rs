#!/usr/bin/env python3
import re
import sys
import logging
from pathlib import Path
from argparse import ArgumentParser


def main(argv=None) -> int:
    parser = ArgumentParser(
        description="Check mapping presence between reference and incremental canonical ids and a rebuild log"
    )
    parser.add_argument(
        "ref", type=Path, help="Reference file containing canonical ids"
    )
    parser.add_argument(
        "incr", type=Path, help="Incremental file containing canonical ids"
    )
    parser.add_argument(
        "log", type=Path, help="Rebuild log to search for NodeId entries"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=r':canonical-id "(\d+v\d+)"',
        help="Regex pattern to extract canonical ids (one capture group)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=60,
        help="Number of characters of surrounding context to show from the log",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )

    # validate files
    for p in (args.ref, args.incr, args.log):
        if not p.exists():
            logging.error("File not found: %s", p)
            return 2
        if not p.is_file():
            logging.error("Not a file: %s", p)
            return 2

    try:
        id_pattern = re.compile(args.pattern)
    except re.error as e:
        logging.error("Invalid regex pattern: %s -> %s", args.pattern, e)
        return 2

    logging.info(
        "Checking mapping:\n Reference - %s\n Incremental - %s\n Log - %s\n Pattern - %s",
        args.ref,
        args.incr,
        args.log,
        id_pattern.pattern,
    )

    ref_text = args.ref.read_text(encoding="utf-8", errors="replace")
    incr_text = args.incr.read_text(encoding="utf-8", errors="replace")
    log_text = args.log.read_text(encoding="utf-8", errors="replace")

    ref_ids = id_pattern.findall(ref_text)
    incr_ids = id_pattern.findall(incr_text)

    logging.debug(
        "Found %d ids in reference, %d ids in incremental", len(ref_ids), len(incr_ids)
    )

    # find first index where they differ
    first_diff = None
    for i, (r, iid) in enumerate(zip(ref_ids, incr_ids)):
        if r != iid:
            first_diff = (i, r, iid)
            break

    if first_diff is None:
        if len(ref_ids) != len(incr_ids):
            logging.info(
                "No mismatch in zipped ids, but counts differ: ref=%d incr=%d",
                len(ref_ids),
                len(incr_ids),
            )
            # report the extra ids
            if len(ref_ids) > len(incr_ids):
                logging.info(
                    "First extra id in reference at index %d: %s",
                    len(incr_ids),
                    ref_ids[len(incr_ids)],
                )
            else:
                logging.info(
                    "First extra id in incremental at index %d: %s",
                    len(ref_ids),
                    incr_ids[len(ref_ids)],
                )
            # treat as difference
            return 1
        logging.info("No diff found among zipped ids")
        return 0

    logging.info("first_diff=%s", first_diff)

    # Check presence in log and numeric matches
    _, ref_token, incr_token = first_diff

    def safe_split_num(token: str) -> str:
        if "v" in token:
            return token.split("v", 1)[0]
        return token

    any_found = False
    for token in (incr_token, ref_token):
        present = token in log_text
        logging.info("Token %s present in rebuild log? %s", token, present)
        num = safe_split_num(token)
        nodeids = re.findall(r"NodeId\((\d+v\d+)\)", log_text)
        found_any = any(n.startswith(num + "v") for n in nodeids)
        logging.info("Numeric prefix %s found among NodeIds in log? %s", num, found_any)
        any_found = any_found or present or found_any

    # show some context for numeric matches
    for token in (incr_token, ref_token):
        num = safe_split_num(token)
        logging.info("\n=== Context for numeric prefix %s ===", num)
        for m in re.finditer(r"NodeId\((\d+v\d+)\)", log_text):
            if m.group(1).startswith(num + "v"):
                start = max(0, m.start() - args.context)
                end = min(len(log_text), m.end() + args.context)
                snippet = log_text[start:end].replace("\n", "\\n")
                logging.info("...\n%s", snippet)
                break

    # exit code: 0 = no diffs, 1 = diff(s) found, 3 = diff but not mentioned in log
    if any_found:
        return 1
    return 3


if __name__ == "__main__":
    sys.exit(main())
