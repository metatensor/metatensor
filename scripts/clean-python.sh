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

find . -name "*.egg-info" -exec rm -rf "{}" +
find . -name "__pycache__" -exec rm -rf "{}" +
find . -name ".coverage" -exec rm -rf "{}" +
