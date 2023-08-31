#!/usr/bin/env bash

# This script creates an archive containing the sources for the C++ part of
# metatensor-torch, and copy it to be included in the metatensor-torch python
# package sdist.

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
set -eux

cd $ROOT_DIR
tar cf metatensor-torch.tar metatensor-torch
gzip -9 metatensor-torch.tar

mv metatensor-torch.tar.gz python/metatensor-torch/
