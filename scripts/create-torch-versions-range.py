#!/usr/bin/env python3
"""
This script updates the `Requires-Dist` information in metatensor-torch wheel METADATA
to contain the range of compatible torch versions. It expects newline separated
`Requires-Dist: torch ==...` information (corresponding to wheels built against a single
torch version) and will print `Requires-Dist: torch >=$MIN_VERSION,<${MAX_VERSION+1}` on
the standard output.

This output can the be used in the merged wheel containing the build against all torch
versions.
"""
import re
import sys


if __name__ == "__main__":
    torch_versions_raw = sys.argv[1]

    torch_versions = []
    for version in torch_versions_raw.split("\n"):
        if version.strip() == "":
            continue

        match = re.match(r"Requires-Dist: torch[ ]?==(\d+)\.(\d+)\.\*", version)
        if match is None:
            raise ValueError(f"unexpected Requires-Dist format: {version}")

        major, minor = match.groups()
        major = int(major)
        minor = int(minor)

        version = (major, minor)

        if version in torch_versions:
            raise ValueError(f"duplicate torch version: {version}")

        torch_versions.append(version)

    torch_versions = list(sorted(torch_versions))

    min_version = f"{torch_versions[0][0]}.{torch_versions[0][1]}"
    max_version = f"{torch_versions[-1][0]}.{torch_versions[-1][1] + 1}"

    print(f"Requires-Dist: torch >={min_version},<{max_version}")
