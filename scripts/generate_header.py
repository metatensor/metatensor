#!/usr/bin/env python3
"""
Amalgamates C++ header files into a single header by recursively expanding
#include directives in-place.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Set


LOG = logging.getLogger(__name__)

# This set will track which files have already been included to prevent duplicates.
PROCESSED_FILES: Set[Path] = set()


def setup_logging() -> None:
    """Configure the logging format and level."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def expand_includes_recursively(
    file_path: Path,
    output_stream,
    project_include_dir: Path,
):
    """
    Reads a file, writes its content to the stream, and recursively expands
    any internal #include directives it finds.
    """
    resolved_path = file_path.resolve()
    if not resolved_path.exists():
        LOG.critical("Included file does not exist: %s", file_path)
        sys.exit(1)

    if resolved_path in PROCESSED_FILES:
        return  # This file's content has already been included.

    PROCESSED_FILES.add(resolved_path)
    LOG.info("Expanding %s", resolved_path.relative_to(project_include_dir.parent))

    lib_relative_path = resolved_path.relative_to(project_include_dir).as_posix()
    output_stream.write(f"// --- Start of {lib_relative_path} ---\n\n")

    # This pattern now captures the delimiter (< or ") as the first group.
    include_pattern = re.compile(r'#\s*include\s*([<"])([^>"]+)([>"])')

    with open(resolved_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip() == "#pragma once":
                continue

            match = include_pattern.match(line.strip())
            if not match:
                # Not an include line, just write it out.
                output_stream.write(line)
                continue

            # We found an include directive.
            delimiter = match.group(1)
            include_target_str = match.group(2)

            if delimiter == "<":
                # It's a system header (e.g. <vector>), so we always keep it.
                output_stream.write(line)
                continue

            # It's a quoted include "...", so we check if it's an internal file.
            potential_path = (resolved_path.parent / include_target_str).resolve()

            # This is always installed
            if include_target_str == "metatensor.h":
                # So we also keep it
                output_stream.write(line)
                continue

            is_internal = False
            try:
                # An include is internal if its resolved path is inside the project's
                # main include directory.
                potential_path.relative_to(project_include_dir)
                is_internal = True
            except ValueError:
                is_internal = False

            if is_internal and potential_path.exists():
                # It's an internal header we need to expand recursively.
                expand_includes_recursively(
                    potential_path, output_stream, project_include_dir
                )
            else:
                # It's an external local include (like "metatensor.h") or a file
                # not found in the project source, so we keep the include directive.
                output_stream.write(line)

    output_stream.write(f"\n// --- End of {lib_relative_path} ---\n\n")


def main() -> None:
    """The main execution function for the script."""
    parser = argparse.ArgumentParser(
        description="Recursively expand includes to create a single header file."
    )
    parser.add_argument(
        "entry_point",
        type=Path,
        help="The main 'table of contents' header file to start processing from.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path for the final generated header file.",
    )
    args = parser.parse_args()
    setup_logging()

    # The project include dir is the parent of the directory containing the entry point
    # e.g., if entry is '.../include/metatensor/metatensor.hpp', this is '.../include'
    project_include_dir = args.entry_point.resolve().parent.parent

    try:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as outfile:
            outfile.write("// Amalgamated header for metatensor\n")
            outfile.write("// Generated automatically by the build system\n\n")

            # Start the recursive expansion from the main entry point file.
            expand_includes_recursively(
                args.entry_point,
                outfile,
                project_include_dir,
            )
    except Exception as e:
        LOG.critical("Failed to generate amalgamated header: %s", e)
        sys.exit(1)

    LOG.info("Successfully generated amalgamated header: %s", args.output_file)


if __name__ == "__main__":
    main()
