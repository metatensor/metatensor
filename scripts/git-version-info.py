#!/usr/bin/env python3
"""
This script calls git to get the number of commits since the last tag, as well as a full
hash of all code. It then prints both to stdout.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def warn_and_exit(message, exit_code=0):
    print(message, file=sys.stderr)
    print(0, file=sys.stdout)
    sys.exit(exit_code)


def run_subprocess(args, check=True, env=None):
    output = subprocess.run(
        args,
        capture_output=True,
        encoding="utf8",
        check=False,
        env=env,
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


def n_commits_since_last_tag(tag_prefix):
    # get the full list of tags
    result = run_subprocess(["git", "tag", "--sort=-creatordate"])
    all_tags = result.stdout.strip().split("\n")

    latest_tag = None
    for tag in all_tags:
        if not tag:
            continue

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

    return n_commits


def git_hash_all_code():
    # make sure the index is up to date before doing `git diff-index`
    run_subprocess(["git", "update-index", "-q", "--really-refresh"])

    output = subprocess.run(
        ["git", "diff-index", "--quiet", "HEAD", "--"],
        capture_output=True,
    )

    if output.returncode not in [0, 1]:
        raise Exception(
            "failed to get git information (`git diff-index`).\n"
            f"stdout: {output.stdout}\n"
            f"stderr: {output.stderr}\n"
        )

    is_dirty = output.returncode == 1
    if is_dirty:
        # This gets the full git hash for the current repo, including non-committed and
        # non-staged code (cf https://stackoverflow.com/a/48213033). It does so by
        # pretending to stage all the file using a temporary index. This way the actual
        # git index is left untouched.
        with tempfile.NamedTemporaryFile("wb") as tmp:
            with open(os.path.join(ROOT, ".git", "index"), "rb") as git_index:
                shutil.copyfileobj(git_index, tmp)
            tmp.close()

            git_env = os.environ.copy()
            git_env["GIT_INDEX_FILE"] = tmp.name
            run_subprocess(["git", "add", "--all"], env=git_env)

            output = run_subprocess(["git", "write-tree"], env=git_env)
            short_hash = output.stdout[:7]
    else:
        output = run_subprocess(["git", "rev-parse", "HEAD"])
        short_hash = output.stdout[:7]

    return ("dirty." if is_dirty else "git.") + short_hash


if __name__ == "__main__":
    if len(sys.argv) != 2:
        warn_and_exit(f"usage: {sys.argv[0]} <tag-prefix>", exit_code=1)

    tag_prefix = sys.argv[1]

    try:
        result = run_subprocess(["git", "--version"])
    except Exception:
        warn_and_exit("could not run `git --version`, is git installed on your system?")

    # We need git >=2.0 for `git tag --sort=-creatordate`
    _, _, git_version, *_ = result.stdout.split()
    if not re.match(r"2\.\d+\.\d+", git_version):
        warn_and_exit(f"this script requires git>=2.0, we found git v{git_version}")

    result = run_subprocess(["git", "rev-parse", "--show-toplevel"], check=False)

    if result.returncode != 0 or not os.path.samefile(result.stdout.strip(), ROOT):
        warn_and_exit(
            "the git root is not featomic repository, if you are trying to build "
            "featomic from source please use a git checkout"
        )

    print(n_commits_since_last_tag(tag_prefix))
    print(git_hash_all_code())
