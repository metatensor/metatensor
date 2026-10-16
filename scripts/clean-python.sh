#!/usr/bin/env bash

# This script removes all temporary files created by Python during
# installation and tests running.

set -eux

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

rm -rf dist
rm -rf build
rm -rf docs/build
rm -rf docs/src/examples
rm -rf docs/src/sg_execution_times.rst

rm -rf python/metatensor_core/metatensor-core-cxx-*.tar.gz
rm -rf python/metatensor_core/dist
rm -rf python/metatensor_core/build

rm -rf python/metatensor_operations/dist
rm -rf python/metatensor_operations/build

rm -rf python/metatensor_torch/metatensor-torch-cxx-*.tar.gz
rm -rf python/metatensor_torch/dist
rm -rf python/metatensor_torch/build

rm -rf python/metatensor_learn/dist
rm -rf python/metatensor_learn/build

set +x

# Remove `__pycache__`, `.egg-info`, and `.coverage` files everywhere, except in
# `.tox` and in directories ignored by the user's git configuration (global
# excludes file and `.git/info/exclude`, but not this repository's .gitignore).
GIT_EXCLUDES=(--exclude=.tox/)
GLOBAL_EXCLUDES_FILE=$(git config --path core.excludesFile || echo "${XDG_CONFIG_HOME:-$HOME/.config}/git/ignore")
for file in "$GLOBAL_EXCLUDES_FILE" "$(git rev-parse --git-path info/exclude)"; do
    if [ -f "$file" ]; then
        GIT_EXCLUDES+=(--exclude-from="$file")
    fi
done

# With `--directory`, fully ignored directories are listed once
FIND_PRUNE=(-name .git)
while IFS= read -r -d '' path; do
    if [[ "$path" == */ ]]; then
        FIND_PRUNE+=(-o -path "./${path%/}")
    fi
done < <(git ls-files -z --others --ignored --directory "${GIT_EXCLUDES[@]}")

set -x

find . \( "${FIND_PRUNE[@]}" \) -prune -o \
    \( -name "__pycache__" -o -name "*.egg-info" -o -name ".coverage" \) -prune \
    -exec rm -rf "{}" +
