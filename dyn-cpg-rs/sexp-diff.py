#!/usr/bin/env python3
import sys
import tempfile
import subprocess
import os
import re


def preprocess_file(path):
    """Load file and strip :id patterns"""
    try:
        with open(path, "r") as f:
            content = f.read()

        content = re.sub(r':id\s+"[^"]*"', "", content)

        return content

    except FileNotFoundError:
        print(f"Error: File '{path}' not found", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error processing '{path}': {e}", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("Usage: cpg-diff [diff-options] file1 file2", file=sys.stderr)
        sys.exit(2)

    # Parse arguments - options that take parameters need special handling
    files = []
    diff_options = []
    args = sys.argv[1:]
    i = 0

    while i < len(args):
        arg = args[i]
        if arg.startswith("-"):
            diff_options.append(arg)
            # Check if this option takes a parameter
            if arg in ["-W", "--width", "-L", "--label"] and i + 1 < len(args):
                i += 1
                diff_options.append(args[i])
            elif (
                "=" not in arg
                and arg.startswith("--")
                and i + 1 < len(args)
                and not args[i + 1].startswith("-")
            ):
                # Handle --option value format
                i += 1
                diff_options.append(args[i])
        else:
            files.append(arg)
        i += 1

    if len(files) != 2:
        print(
            f"Error: Expected exactly 2 files, got {len(files)}: {files}",
            file=sys.stderr,
        )
        sys.exit(2)

    old_path, new_path = files

    # Verify files exist
    if not os.path.exists(old_path):
        print(f"Error: '{old_path}' does not exist", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(new_path):
        print(f"Error: '{new_path}' does not exist", file=sys.stderr)
        sys.exit(2)

    # Process files
    old_processed = preprocess_file(old_path)
    new_processed = preprocess_file(new_path)

    # Create temp files and ensure cleanup
    temp_old = None
    temp_new = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sexp", delete=False
        ) as f_old:
            temp_old = f_old.name
            f_old.write(old_processed)
            f_old.flush()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sexp", delete=False
        ) as f_new:
            temp_new = f_new.name
            f_new.write(new_processed)
            f_new.flush()

        # Run diff with original filenames in output
        diff_cmd = ["diff"] + diff_options + [temp_old, temp_new]

        # Replace temp filenames with original names in diff output
        result = subprocess.run(diff_cmd, capture_output=True, text=True)

        if result.stdout:
            output = result.stdout.replace(temp_old, old_path).replace(
                temp_new, new_path
            )
            print(output, end="")

        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")

        sys.exit(result.returncode)

    finally:
        # Clean up temp files
        for temp_file in [temp_old, temp_new]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()
