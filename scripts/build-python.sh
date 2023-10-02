#!/usr/bin/env bash

set -eux

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

TMP_DIR="$1"
rm -rf "$TMP_DIR"/dist

./scripts/package-core.sh
./scripts/package-torch.sh

# check building sdist from a checkout, and wheel from the sdist
python -m build python/metatensor-core --outdir "$TMP_DIR"/dist
python -m build python/metatensor-operations --outdir "$TMP_DIR"/dist
python -m build . --outdir "$TMP_DIR"/dist

# for metatensor-torch, we need a pre-built version of metatensor-core, so
# we use the one we just generated and make it available to pip
dir2pi --no-symlink "$TMP_DIR"/dist

PORT=8912
if nc -z localhost $PORT; then
    printf "\033[91m ERROR: an application is listening to port %d. Please free up the port first. \033[0m\n" $PORT >&2
    exit 1
fi

PYPI_SERVER_PID=""
function cleanup() {
    kill $PYPI_SERVER_PID
}
# Make sure to stop the Python server on script exit/cancellation
trap cleanup INT TERM EXIT

python -m http.server --directory "$TMP_DIR"/dist $PORT &
PYPI_SERVER_PID=$!

# add the python server to the set of extra pip index URL
export PIP_EXTRA_INDEX_URL="http://localhost:$PORT/simple/ ${PIP_EXTRA_INDEX_URL=}"

# build metatensor-torch, using metatensor-core from `PIP_EXTRA_INDEX_URL`
# for the sdist => wheel build.
python -m build python/metatensor-torch --outdir "$TMP_DIR/dist"