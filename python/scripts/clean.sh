#/usr/bin/env bash

set -eux

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH/../..

rm -rf dist
rm -rf build

cd $SCRIPTPATH/..
rm -rf src/equistore/lib/lib*
rm -rf src/*.egg-info
