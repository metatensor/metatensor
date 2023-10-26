#!/usr/bin/env python3
"""
This script calls git to get the number of commits since the last tag matching a
pattern (pattern given on the command line).
"""
import os
import subprocess
import sys


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def warn_and_exit(message, exit_code=0):
    print(message, file=sys.stderr)
    print(0, file=sys.stdout)
    sys.exit(exit_code)


if __name__ == "__main__":
    output = subprocess.run(
        ["git", "--version"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        check=False,
    )

    if output.returncode != 0:
        warn_and_exit("could not run `git`, is it installed on your system?")

    if len(sys.argv) != 2:
        warn_and_exit(f"usage: {sys.argv[0]} <tag-prefix>", exit_code=1)

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        check=False,
        encoding="utf8",
    )

    if result.returncode != 0 or not os.path.samefile(result.stdout.strip(), ROOT):
        warn_and_exit(
            "the git root is not metatensor repository, if you are trying to build "
            "metatensor from source please use a git checkout"
        )

    tag_prefix = sys.argv[1]

    # get the full list of tags
    result = subprocess.run(
        ["git", "tag", "--sort=-creatordate"],
        capture_output=True,
        check=True,
        encoding="utf8",
    )
    all_tags = result.stdout.strip().split("\n")

    latest_tag = None
    for tag in all_tags:
        if tag.startswith(tag_prefix):
            latest_tag = tag
            break

    if latest_tag is None:
        # no matching tags, use the first commit as the original ref
        result = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            capture_output=True,
            check=True,
            encoding="utf8",
        )
        reference = result.stdout.strip()
    else:
        # get the commit corresponding to the most recent tag
        result = subprocess.run(
            ["git", "rev-parse", f"{latest_tag}^0"],
            capture_output=True,
            check=True,
            encoding="utf8",
        )
        reference = result.stdout.strip()

    result = subprocess.run(
        ["git", "rev-list", f"{reference}..HEAD", "--count"],
        capture_output=True,
        check=True,
        encoding="utf8",
    )
    n_commits = result.stdout.strip()

    print(n_commits)
