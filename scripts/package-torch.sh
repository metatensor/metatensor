#!/usr/bin/env bash

# This script creates an archive containing the sources for the C++ part of
# equistore-torch, and copy it to be included in the equistore-torch python
# package sdist.

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
set -eux

cd $ROOT_DIR
tar cf equistore-torch.tar equistore-torch
gzip -9 equistore-torch.tar

mv equistore-torch.tar.gz python/equistore-torch/
