#!/usr/bin/env python3
"""
This script calls git to get the number of commits since the last tag matching a
pattern (pattern given on the command line).
"""
import os
import re
import subprocess
import sys


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def warn_and_exit(message, exit_code=0):
    print(message, file=sys.stderr)
    print(0, file=sys.stdout)
    sys.exit(exit_code)


def run_subprocess(args, check=True):
    output = subprocess.run(
        args,
        capture_output=True,
        encoding="utf8",
        check=False,
    )

    if check and output.returncode != 0:
        raise Exception(
            f"failed to run '{' '.join(args)}' (exit code {output.returncode})\n"
            f"stdout: {output.stdout}\n"
            f"stderr: {output.stderr}\n"
        )

    if output.stderr != "":
        print(output.stderr, file=sys.stderr)

    return output


if __name__ == "__main__":
    if len(sys.argv) != 2:
        warn_and_exit(f"usage: {sys.argv[0]} <tag-prefix>", exit_code=1)

    try:
        result = run_subprocess(["git", "--version"])
    except Exception:
        warn_and_exit("could not run `git --version`, is git installed on your system?")

    # We need git >=2.0 for `git tag --sort=-creatordate` below
    _, _, git_version, *_ = result.stdout.split()
    if not re.match(r"2\.\d+\.\d+", git_version):
        warn_and_exit(f"this script requires git>=2.0, we found git v{git_version}")

    result = run_subprocess(["git", "rev-parse", "--show-toplevel"], check=False)

    if result.returncode != 0 or not os.path.samefile(result.stdout.strip(), ROOT):
        warn_and_exit(
            "the git root is not metatensor repository, if you are trying to build "
            "metatensor from source please use a git checkout"
        )

    tag_prefix = sys.argv[1]

    # get the full list of tags
    result = run_subprocess(["git", "tag", "--sort=-creatordate"])
    all_tags = result.stdout.strip().split("\n")

    latest_tag = None
    for tag in all_tags:
        if tag.startswith(tag_prefix):
            latest_tag = tag
            break

    if latest_tag is None:
        # no matching tags, use the first commit as the original ref
        result = run_subprocess(["git", "rev-list", "--max-parents=0", "HEAD"])
        reference = result.stdout.strip()
    else:
        # get the commit corresponding to the most recent tag
        result = run_subprocess(["git", "rev-parse", f"{latest_tag}^0"])
        reference = result.stdout.strip()

    result = run_subprocess(["git", "rev-list", f"{reference}..HEAD", "--count"])
    n_commits = result.stdout.strip()

    print(n_commits)
