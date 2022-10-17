#/usr/bin/env bash

set -eux

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH/..

rm -rf *.egg-info
rm -rf dist build
rm -rf equistore/lib/lib*
