#!/usr/bin/env bash

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/../.. && pwd)
set -eux

rm -rf $ROOT_DIR/target/package
cd $ROOT_DIR/equistore-core

# Package equistore-core and copy the .crate file to equistore/core/
cargo package --allow-dirty --no-verify
cp $ROOT_DIR/target/package/equistore-core-*.crate $ROOT_DIR/equistore/

# Regenerate Rust bindings to the C API
cd $ROOT_DIR/equistore

bindgen $ROOT_DIR/equistore-core/include/equistore.h -o src/c_api.rs \
    --disable-header-comment \
    --default-macro-constant-type=signed \
    --merge-extern-blocks \
    --allowlist-function "^eqs_.*" \
    --allowlist-type "^eqs_.*" \
    --allowlist-var "^EQS_.*"\
    --must-use-type "eqs_status_t" \
    --raw-line '#![allow(warnings)]
//! Rust definition corresponding to equistore-core C-API.
//!
//! This module is exported for advanced users of the equistore crate, but
//! should not be needed by most.

#[cfg_attr(feature="static", link(name="equistore", kind = "static", modifiers = "-whole-archive"))]
#[cfg_attr(not(feature="static"), link(name="equistore", kind = "dylib"))]
extern "C" {}'
