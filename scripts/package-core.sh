#!/usr/bin/env bash

# This script creates an archive containing the sources for the metatensor-core
# Rust crate, and copy it to be included in the metatensor crate source release,
# and the metatensor-core python package sdist.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
set -eux

rm -rf "$ROOT_DIR/target/package"
cd "$ROOT_DIR/metatensor-core"

# Package metatensor-core using cargo tools, and add a file for
# `n_commits_since_last_tag`
cargo package --allow-dirty --no-verify

TMP_DIR=$(mktemp -d)

cd "$TMP_DIR"
tar xf "$ROOT_DIR"/target/package/metatensor-core-*.crate
ARCHIVE_NAME=$(ls)
ARCHIVE_NAME="metatensor-core-cxx-${ARCHIVE_NAME:16}"

mv metatensor-core-* "$ARCHIVE_NAME"

# Get the number of commits since last tag
cd "$ROOT_DIR"
./scripts/n-commits-since-last-tag.py "metatensor-core-v" > "$TMP_DIR/$ARCHIVE_NAME/cmake/n_commits_since_last_tag"

cd "$TMP_DIR"
tar cf "$ARCHIVE_NAME.tar" "$ARCHIVE_NAME"
gzip -9 "$ARCHIVE_NAME.tar"

cp "$TMP_DIR/$ARCHIVE_NAME.tar.gz" "$ROOT_DIR/metatensor/"
cp "$TMP_DIR/$ARCHIVE_NAME.tar.gz" "$ROOT_DIR/python/metatensor-core/"

mkdir -p "$ROOT_DIR/dist/cxx"
cp "$TMP_DIR/$ARCHIVE_NAME.tar.gz" "$ROOT_DIR/dist/cxx/"
