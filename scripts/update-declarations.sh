#!/usr/bin/env bash

# This script update the declaration corresponding to metatensor-core C-API
# in the metatensor Rust sources (`bindgen` below) and the metatensor-core
# Python package

ROOT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
set -eux

cd $ROOT_DIR

$ROOT_DIR/scripts/package-core.sh

# Regenerate Rust bindings to the C API
bindgen $ROOT_DIR/metatensor-core/include/metatensor.h -o $ROOT_DIR/metatensor/src/c_api.rs \
    --disable-header-comment \
    --no-doc-comments \
    --default-macro-constant-type=signed \
    --merge-extern-blocks \
    --allowlist-function "^mts_.*" \
    --allowlist-type "^mts_.*" \
    --allowlist-var "^MTS_.*"\
    --must-use-type "mts_status_t" \
    --raw-line '#![allow(warnings)]
//! Rust definition corresponding to metatensor-core C-API.
//!
//! This module is exported for advanced users of the metatensor crate, but
//! should not be needed by most.

#[cfg_attr(feature="static", link(name="metatensor", kind = "static", modifiers = "-whole-archive"))]
#[cfg_attr(all(not(feature="static"), not(target_os="windows")), link(name="metatensor", kind = "dylib"))]
#[cfg_attr(all(not(feature="static"), target_os="windows"), link(name="metatensor.dll", kind = "dylib"))]
extern "C" {}'


# Regenerate Python bindings to the C API
$ROOT_DIR/python/scripts/generate-declarations.py
