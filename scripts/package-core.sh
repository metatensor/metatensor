#!/usr/bin/env bash

# This script creates an archive containing the sources for the equistore-core
# Rust crate, and copy it to be included in the equistore crate source release,
# and the equistore-core python package sdist.

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
set -eux

rm -rf $ROOT_DIR/target/package
cd $ROOT_DIR/equistore-core

# Package equistore-core and copy the .crate file
cargo package --allow-dirty --no-verify
cp $ROOT_DIR/target/package/equistore-core-*.crate $ROOT_DIR/equistore/
cp $ROOT_DIR/target/package/equistore-core-*.crate $ROOT_DIR/python/equistore-core/
