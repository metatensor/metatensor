# Changelog

All notable changes to the metatensor Rust crate are documented here, following
the [keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/lab-cosmo/metatensor/)
<!--
### Added

### Fixed

### Changed

### Removed
-->

## [Version 0.1.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-rust-v0.1.0) - 2023-10-11

### Added

- Rust bindings to all metatensor-core class: `Labels`, `LabelsEntry`,
  `TensorBlock`, and `TensorMap`;
- Rust binding of `mts_array_t` through the `Array` trait;
- Implementation of `Array` for [ndarray](https://docs.rs/ndarray/);
- Parallel iteration over labels entries and blocks using [rayon](https://docs.rs/rayon/);
