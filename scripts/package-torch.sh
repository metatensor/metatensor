#!/usr/bin/env bash

# This script creates an archive containing the sources for the C++ part of
# metatensor-torch, and copy it to be included in the metatensor-torch python
# package sdist.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
set -eux

VERSION=$(cat "$ROOT_DIR/metatensor-torch/VERSION")
ARCHIVE_NAME="metatensor-torch-cxx-$VERSION"

TMP_DIR=$(mktemp -d)
mkdir "$TMP_DIR/$ARCHIVE_NAME"

cp -r "$ROOT_DIR"/metatensor-torch/* "$TMP_DIR/$ARCHIVE_NAME/"
cp "$ROOT_DIR/LICENSE" "$TMP_DIR/$ARCHIVE_NAME"
cp "$ROOT_DIR/AUTHORS" "$TMP_DIR/$ARCHIVE_NAME"

# Get the number of commits since last tag, this is used when building the
# code to change the version for development builds
cd "$ROOT_DIR"
./scripts/n-commits-since-last-tag.py "metatensor-torch-v" > "$TMP_DIR/$ARCHIVE_NAME/cmake/n_commits_since_last_tag"

cd "$TMP_DIR"
tar cf "$ARCHIVE_NAME".tar "$ARCHIVE_NAME"

gzip -9 "$ARCHIVE_NAME".tar
cp "$ARCHIVE_NAME".tar.gz "$ROOT_DIR/python/metatensor-torch/"

mkdir -p "$ROOT_DIR/dist/cxx"
cp "$ARCHIVE_NAME".tar.gz "$ROOT_DIR/dist/cxx/"
