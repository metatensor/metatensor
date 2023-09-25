#!/usr/bin/env bash

# This script creates an archive containing the sources for the metatensor-core
# Rust crate, and copy it to be included in the metatensor crate source release,
# and the metatensor-core python package sdist.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
set -eux

rm -rf "$ROOT_DIR/target/package"
cd "$ROOT_DIR/metatensor-core"

# Package metatensor-core and copy the .crate file
cargo package --allow-dirty --no-verify
cp "$ROOT_DIR"/target/package/metatensor-core-*.crate "$ROOT_DIR/metatensor/"
cp "$ROOT_DIR"/target/package/metatensor-core-*.crate "$ROOT_DIR/python/metatensor-core/"
