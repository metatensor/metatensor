#!/usr/bin/env bash

# This script creates an archive containing the sources for the C++ part of
# metatensor-torch, and copy it to be included in the metatensor-torch python
# package sdist.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
set -eux

cd "$ROOT_DIR"

VERSION=$(cat metatensor-torch/VERSION)
ARCHIVE_NAME="metatensor-torch-cxx-$VERSION"

./scripts/n-commits-since-last-tag.py "metatensor-torch-v" > metatensor-torch/cmake/n_commits_since_last_tag
tar cf "$ARCHIVE_NAME".tar metatensor-torch
rm -f metatensor-torch/cmake/n_commits_since_last_tag

gzip -9 "$ARCHIVE_NAME".tar
cp "$ARCHIVE_NAME".tar.gz "$ROOT_DIR/python/metatensor-torch/"

mkdir -p "$ROOT_DIR/dist/cxx"
mv "$ARCHIVE_NAME".tar.gz "$ROOT_DIR/dist/cxx/"
