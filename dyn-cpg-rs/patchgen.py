#!/usr/bin/env python3
"""
AST-driven base/patch generator for C code, designed for incremental parsing tests.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import random

from tree_sitter import Language, Parser, Node
import tree_sitter_c as ts_c


# ------------------------- Tree-sitter setup -------------------------
C_LANG = Language(ts_c.language())
C_PARSER = Parser(C_LANG)


# --------------------------- Data models ----------------------------
@dataclass
class RemovedStmt:
    text: str
    func_name: Optional[str]
    # Byte range of the removed node in the *version it was removed from* (for reference)
    start_byte: int
    end_byte: int


@dataclass
class GenConfig:
    num_patches: int
    base_removals: int
    seed: int
    validate_cmd: Optional[str]
    verbose: bool


# --------------------------- Utilities ------------------------------

PATCH_TEMPLATE = "{num:03d}_{name}_{info}.patch"


def parse(source: str):
    return C_PARSER.parse(source.encode("utf-8"))


def source_from_bytes(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def slice_replace(src: str, start: int, end: int, replacement: str) -> str:
    return src[:start] + replacement + src[end:]


def trim_trailing_ws(s: str) -> str:
    return re.sub(r"[ \t]+(?=\n)", "", s)


# --------------------------- AST helpers ----------------------------

# Statement-ish node types we consider safe to remove/insert around
STMT_TYPES = {
    "expression_statement",
    "declaration",
    "if_statement",
    "for_statement",
    "while_statement",
    "return_statement",
}


def iter_functions(root: Node) -> List[Node]:
    return [n for n in root.children if n.type == "function_definition"]


def function_name(fn_node: Node, src: bytes) -> Optional[str]:
    # function_definition -> (declaration_specifiers?) function_declarator compound_statement
    # for ch in fn_node.children:
    #     if ch.type == "function_declarator":
    #         for g in ch.children:
    #             if g.type == "identifier":
    #                 return src[g.start_byte : g.end_byte].decode("utf-8")
    # return None

    decl = fn_node.child_by_field_name("declarator")
    if not decl:
        return None

    if decl.type == "pointer_declarator":
        decl = decl.child_by_field_name("declarator")
        if not decl:
            return None

    ident = None
    for ch in decl.children:
        if ch.type == "identifier":
            ident = ch
            break
    if not ident:
        return None

    return src[ident.start_byte : ident.end_byte].decode("utf-8", "replace")


def function_body_node(fn_node: Node) -> Optional[Node]:
    for ch in fn_node.children:
        if ch.type == "compound_statement":
            return ch
    return None


def collect_statements_in_body(body: Node) -> List[Node]:
    out = []
    stack = [body]
    while stack:
        n = stack.pop()
        if n is not body and n.type in STMT_TYPES:
            out.append(n)
        stack.extend(n.children)
    # Sort by start_byte to get stable order
    out.sort(key=lambda n: n.start_byte)
    return out


def is_safe_removal(stmt: Node, src: bytes) -> bool:
    """
    Heuristics: avoid removing declarations that declare multiple vars, avoid removing return in main
    and don't remove preprocessor or function headers. Only within compound_statement.
    """
    # Disallow removing preprocessor lines entirely (not a statement anyway)
    # Disallow removing if it spans braces of the body start/end
    text = src[stmt.start_byte : stmt.end_byte].decode("utf-8", "replace")
    # Skip declarations with commas (multiple declarators) to keep compile sanity
    if stmt.type == "declaration" and "," in text:
        return False
    # Reject removing case/default labels
    if re.match(r"\s*(case|default)\b", text):
        return False
    return True


# --------------------------- Diff / Files ---------------------------


def unified_diff(old: str, new: str, old_label: str, new_label: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f1:
        f1.write(old)
        p1 = f1.name
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f2:
        f2.write(new)
        p2 = f2.name
    try:
        proc = subprocess.run(
            ["diff", "-u", "--label", old_label, "--label", new_label, p1, p2],
            text=True,
            capture_output=True,
        )
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"diff failed: {proc.stderr}")
        return proc.stdout
    finally:
        os.unlink(p1)
        os.unlink(p2)


def write_text(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


# --------------------------- Validators -----------------------------


def validate_c_source(content: str, validate_cmd: Optional[str], tmp_dir: Path) -> None:
    if not validate_cmd:
        return
    src = tmp_dir / "_validate.c"
    write_text(src, content)
    # The command may include spaces/args; let shell parse for simplicity.
    # We intentionally pass through the filename at the end if no %s placeholder is present.
    cmdline = validate_cmd
    if "%s" in cmdline:
        cmdline = cmdline.replace("%s", str(src))
        args = cmdline
    else:
        args = f"{cmdline} {src}"
    proc = subprocess.run(args, shell=True, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "validation failed (nonzero exit)\n" + proc.stdout + proc.stderr
        )


# --------------------------- Patch ops ------------------------------


class PatchOp:
    def apply(self, src: str, rng: random.Random) -> Tuple[str, str]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        # CamelCaseClassName -> kebab-case-class-name
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1-\2", self.__class__.__name__)
        return re.sub("([a-z0-9])([A-Z])", r"\1-\2", s1).lower()


class DeleteStmt(PatchOp):
    def apply(self, src: str, rng: random.Random) -> Tuple[str, str]:
        tree = parse(src)
        b = src.encode("utf-8")
        root = tree.root_node
        funcs = iter_functions(root)
        candidates: List[Tuple[Node, str]] = []
        for fn in funcs:
            body = function_body_node(fn)
            if not body:
                continue
            stmts = collect_statements_in_body(body)
            name = function_name(fn, b)
            for s in stmts:
                if is_safe_removal(s, b):
                    candidates.append((s, name or "<anon>"))
        if not candidates:
            raise RuntimeError("no removable statements found")
        stmt, fname = rng.choice(candidates)
        new_src = slice_replace(src, stmt.start_byte, stmt.end_byte, "")
        info = f"fn-'{fname}'-delete-from-b{stmt.start_byte}"
        return new_src, info


class InsertDecl(PatchOp):
    def __init__(self, text_factory=None):
        self.text_factory = text_factory or (
            lambda idx: f"int gen_var_{idx} = {idx};\n"
        )

    def apply(self, src: str, rng: random.Random) -> Tuple[str, str]:
        tree = parse(src)
        b = src.encode("utf-8")
        funcs = iter_functions(tree.root_node)
        if not funcs:
            raise RuntimeError("no function to insert into")
        fn = rng.choice(funcs)
        body = function_body_node(fn)
        if not body:
            raise RuntimeError("function has no body")
        # Insert right after the opening brace of the body
        insert_at = body.start_byte + 1  # after '{'
        # Find a reasonable indentation (same line indentation)
        line_start = src.rfind("\n", 0, insert_at) + 1
        indent = re.match(r"[ \t]*", src[line_start:insert_at]).group(0) + "    "
        idx = rng.randint(1, 1_000_000)
        text = indent + self.text_factory(idx)
        new_src = slice_replace(src, insert_at, insert_at, text)
        fname = function_name(fn, b) or "<anon>"
        info = f"fn-'{fname}'-insert-at-{idx}"
        return new_src, info


class RenameLocal(PatchOp):
    def apply(self, src: str, rng: random.Random) -> Tuple[str, str]:
        b = src.encode("utf-8")
        tree = parse(src)
        root = tree.root_node
        funcs = iter_functions(root)
        # Collect candidate (fn, name, ranges)
        candidates: List[Tuple[Node, str, Tuple[int, int]]] = []
        for fn in funcs:
            body = function_body_node(fn)
            if not body:
                continue
            # search for declarations with single identifier
            text = b
            # Walk for declarations
            for n in collect_statements_in_body(body):
                if n.type != "declaration":
                    continue
                decl_text = text[n.start_byte : n.end_byte].decode("utf-8", "replace")
                # single identifier decl like: int foo; or int foo = 0;
                m = re.match(r"\s*\w[\w\s\*]*\s+(\w+)\s*(=|;)", decl_text)
                if not m:
                    continue
                name = m.group(1)
                if any(name.startswith(p) for p in ("gen_var_", "new_")):
                    continue  # avoid generated names
                candidates.append((fn, name, (body.start_byte, body.end_byte)))
        if not candidates:
            raise RuntimeError("no local variables to rename")
        fn, old_name, (lo, hi) = rng.choice(candidates)
        new_name = f"new_{old_name}_{rng.randint(10, 1_000_000)}"
        # Replace occurrences only within function body span, with word boundaries
        region = src[lo:hi]
        # Avoid renaming the function name or types by scoping to body only
        region2 = re.sub(rf"\b{re.escape(old_name)}\b", new_name, region)
        new_src = src[:lo] + region2 + src[hi:]
        fname = function_name(fn, b) or "<anon>"
        info = f"fn-'{fname}'-rename-'{old_name}'-to-'{new_name}'"
        return new_src, info


class ModifyNumLit(PatchOp):
    def apply(self, src: str, rng: random.Random) -> Tuple[str, str]:
        # Avoid includes and macros by skipping lines starting with #
        positions: List[Tuple[int, int, int]] = []  # (start, end, value)
        for m in re.finditer(r"(^[^#].*?\b)(\d+)(\b)", src, flags=re.M):
            start = m.start(2)
            end = m.end(2)
            val = int(m.group(2))
            positions.append((start, end, val))
        if not positions:
            raise RuntimeError("no numeric literals found")
        start, end, val = rng.choice(positions)
        delta = rng.choice([-3, -1, 1, 2, 3, 5])
        new_val = max(0, val + delta)
        new_src = src[:start] + str(new_val) + src[end:]
        info = f"{val}-to-{new_val}"
        return new_src, info


class AddComment(PatchOp):
    def apply(self, src: str, rng: random.Random) -> Tuple[str, str]:
        # Insert a comment before a random function
        tree = parse(src)
        b = src.encode("utf-8")
        funcs = iter_functions(tree.root_node)
        if not funcs:
            # Top-of-file comment fallback
            return "// patch: added comment\n" + src, "add_comment_top"
        fn = rng.choice(funcs)
        # Insert comment at the start of the function_definition
        insert_at = fn.start_byte
        name = function_name(fn, b) or "<anon>"
        text = f"// patch: touch {name}\n"
        # Preserve indentation of the start line
        line_start = src.rfind("\n", 0, insert_at) + 1
        indent = re.match(r"[ \t]*", src[line_start:insert_at]).group(0)
        new_src = slice_replace(src, insert_at, insert_at, indent + text)
        info = f"fn-'{name}'"
        return new_src, info


# ------------------------- Base constructor -------------------------


def build_base(
    source: str, cfg: GenConfig, rng: random.Random
) -> Tuple[str, List[RemovedStmt]]:
    """Remove up to cfg.base_removals 'safe' statements from across functions.
    Deterministic selection order by AST order; we pick every k-th candidate based on seed,
    but always ensure safe removal.
    """
    current = source
    removed: List[RemovedStmt] = []
    attempt = 0
    while len(removed) < cfg.base_removals:
        attempt += 1
        tree = parse(current)
        b = current.encode("utf-8")
        root = tree.root_node
        candidates: List[Tuple[Node, Optional[str]]] = []
        for fn in iter_functions(root):
            body = function_body_node(fn)
            if not body:
                continue
            stmts = collect_statements_in_body(body)
            fname = function_name(fn, b)
            for s in stmts:
                if is_safe_removal(s, b):
                    candidates.append((s, fname))
        if not candidates:
            break
        # Prefer early statements to keep the program compiling; choose with RNG among first N
        window = candidates[: max(1, min(8, len(candidates)))]
        node, fname = rng.choice(window)
        text = b[node.start_byte : node.end_byte].decode("utf-8", "replace")
        current = slice_replace(current, node.start_byte, node.end_byte, "")
        removed.append(
            RemovedStmt(
                text=text,
                func_name=fname,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
            )
        )
        if attempt > 10_000:
            raise RuntimeError("pathological base construction loop")
    return current, removed


# --------------------------- Orchestrator ---------------------------

PATCH_REGISTRY = [
    op()
    for op in locals().values()
    if isinstance(op, type) and issubclass(op, PatchOp) and op is not PatchOp
]


def generate_patches(source_file: Path, out_dir: Path, cfg: GenConfig) -> None:
    src = source_file.read_text(encoding="utf-8")
    if not src.strip():
        raise ValueError("Source file is empty")

    rng = random.Random(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing patches
    for f in out_dir.glob("*.patch"):
        f.unlink()

    # 1) Build base
    base_src, removed_stmts = build_base(src, cfg, rng)
    base_file = out_dir / "base.c"
    write_text(base_file, trim_trailing_ws(base_src))

    if cfg.verbose:
        print(
            f"Base constructed: removed {len(removed_stmts)} statements; wrote {base_file}"
        )

    # 2) Sequential patches
    current = base_src
    tmp_validate_dir = out_dir / ".validate"
    tmp_validate_dir.mkdir(exist_ok=True)

    for i in range(1, cfg.num_patches + 1):
        # rotate registry deterministically
        op = PATCH_REGISTRY[(i - 1) % len(PATCH_REGISTRY)]
        name = op.name
        try:
            new_src, info = op.apply(current, rng)
        except RuntimeError as e:
            # If a particular op fails (no candidates), fallback to InsertDecl to keep sequence going
            if cfg.verbose:
                print(f"op {op.name} failed: {e}; falling back to insert")
            fallback = InsertDecl()
            new_src, info = fallback.apply(current, rng)
            name = fallback.name

        # Validate (optional)
        try:
            validate_c_source(new_src, cfg.validate_cmd, tmp_validate_dir)
        except Exception as ve:
            # If validation fails, try one more safer fallback: AddComment (no semantic impact)
            if cfg.verbose:
                print(
                    f"validation failed for op {op.name}: {ve}; using AddComment fallback"
                )
            safer = AddComment()
            new_src2, info2 = safer.apply(current, rng)
            validate_c_source(new_src2, cfg.validate_cmd, tmp_validate_dir)
            new_src, info, name = new_src2, info2, safer.name

        # Generate diff from current -> new
        diff_txt = unified_diff(
            trim_trailing_ws(current),
            trim_trailing_ws(new_src),
            "base.c" if i == 1 else f"patch{i-1}.c",
            f"patch{i}.c",
        )
        patch_name = out_dir / PATCH_TEMPLATE.format(num=i, name=name, info=info)
        write_text(patch_name, diff_txt)
        if cfg.verbose:
            print(f"Generated {patch_name}")
        current = new_src

    # Cleanup
    if tmp_validate_dir.exists():
        for f in tmp_validate_dir.iterdir():
            f.unlink()
        tmp_validate_dir.rmdir()

    print(f"\nGenerated base.c and {cfg.num_patches} patches in {out_dir}/")


# ------------------------------ CLI --------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate base C file and sequential patches (AST-driven)"
    )
    p.add_argument("source_file", help="Input C source file")
    p.add_argument(
        "-o", "--output-dir", default="seq_patches/tmp", help="Output directory"
    )
    p.add_argument(
        "-n", "--num-patches", type=int, default=3, help="Number of patches to generate"
    )
    p.add_argument(
        "--base-removals",
        type=int,
        default=1,
        help="How many safe statements to remove in base construction",
    )
    p.add_argument(
        "--seed", type=int, default=1337, help="RNG seed for reproducibility"
    )
    p.add_argument(
        "--validate-cmd",
        default=None,
        help=(
            "Optional compiler command to validate each intermediate source. "
            "Example: 'clang -fsyntax-only' or 'gcc -fsyntax-only -std=c11'. "
            "You can use '%s' placeholder to control file position; otherwise the source path is appended."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = p.parse_args(argv)

    sf = Path(args.source_file)
    if not sf.exists():
        print(f"error: source file '{sf}' not found", file=sys.stderr)
        return 2

    out = Path(args.output_dir)
    cfg = GenConfig(
        num_patches=args.num_patches,
        base_removals=max(0, args.base_removals),
        seed=args.seed,
        validate_cmd=args.validate_cmd,
        verbose=args.verbose,
    )

    if not args.validate_cmd:
        print("Warning: No validation command provided; skipping syntax checks.")

    try:
        generate_patches(sf, out, cfg)
    except Exception as e:
        if args.verbose:
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
