#!/usr/bin/env bash

# This script update the declaration corresponding to equistore-core C-API
# in the equistore Rust sources (`bindgen` below) and the equistore-core
# Python package

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
set -eux

cd $ROOT_DIR

$ROOT_DIR/scripts/package-core.sh

# Regenerate Rust bindings to the C API
bindgen $ROOT_DIR/equistore-core/include/equistore.h -o $ROOT_DIR/equistore/src/c_api.rs \
    --disable-header-comment \
    --no-doc-comments \
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
#[cfg_attr(all(not(feature="static"), not(target_os="windows")), link(name="equistore", kind = "dylib"))]
#[cfg_attr(all(not(feature="static"), target_os="windows"), link(name="equistore.dll", kind = "dylib"))]
extern "C" {}'


# Regenerate Python bindings to the C API
$ROOT_DIR/python/scripts/generate-declarations.py
