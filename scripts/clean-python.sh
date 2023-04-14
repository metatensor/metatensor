#/usr/bin/env bash

set -eux

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
cd $ROOT_DIR

rm -rf dist
rm -rf build

rm -rf python/equistore-core/dist
rm -rf python/equistore-core/build

rm -rf python/equistore-operations/dist
rm -rf python/equistore-operations/build

find . -name "*.egg-info" -exec rm -rf "{}" \;
