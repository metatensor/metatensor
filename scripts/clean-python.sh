#/usr/bin/env bash

# This script removes all temporary files created by Python during
# installation and tests running.

set -eux

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
cd $ROOT_DIR

rm -rf dist
rm -rf build

rm -rf python/equistore-core/dist
rm -rf python/equistore-core/build

rm -rf python/equistore-operations/dist
rm -rf python/equistore-operations/build

rm -rf python/equistore-torch/dist
rm -rf python/equistore-torch/build

find . -name "*.egg-info" -exec rm -rf "{}" +
find . -name "__pycache__" -exec rm -rf "{}" +
