#!/usr/bin/env bash

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
set -eux

rm -rf $ROOT_DIR/target/package
cd $ROOT_DIR/equistore-core

# Package equistore-core and copy the .crate file
cargo package --allow-dirty --no-verify
cp $ROOT_DIR/target/package/equistore-core-*.crate $ROOT_DIR/equistore/
cp $ROOT_DIR/target/package/equistore-core-*.crate $ROOT_DIR/python/equistore-core/
