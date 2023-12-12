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

rm -rf python/metatensor-core/metatensor-core-cxx-*.tar.gz
rm -rf python/metatensor-core/dist
rm -rf python/metatensor-core/build

rm -rf python/metatensor-operations/dist
rm -rf python/metatensor-operations/build

rm -rf python/metatensor-torch/metatensor-torch-cxx-*.tar.gz
rm -rf python/metatensor-torch/dist
rm -rf python/metatensor-torch/build

find . -name "*.egg-info" -exec rm -rf "{}" +
find . -name "__pycache__" -exec rm -rf "{}" +
