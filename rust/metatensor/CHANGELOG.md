# Changelog

All notable changes to the metatensor Rust crate are documented here, following
the [keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/metatensor/)
<!--
### Added

### Fixed

### Changed

### Removed
-->

## [Version 0.2.2](https://github.com/metatensor/metatensor/releases/tag/metatensor-rust-v0.2.2) - 2025-07-29

### Added

- `LabelsBuilder::finish_assume_unique` which allows bypassing a check when the
  user can ensure beforehand that all entries will be unique.

## [Version 0.2.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-rust-v0.2.1) - 2025-04-10

### Added

- `Labels::difference` for finding the set difference between two `Labels`

## [Version 0.2.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-rust-v0.2.0) - 2024-09-24

### Changed

- The code now requires Rustc v1.74 to build.
- We now depend on `ndarray=^0.16`

## [Version 0.1.6](https://github.com/metatensor/metatensor/releases/tag/metatensor-rust-v0.1.6) - 2024-09-10

### Added

- Added support for serialization of TensorBlock with `TensorBlock::load`,
  `TensorBlock::load_buffer`, `TensorBlock::save`, `TensorBlock::save_buffer`
  and the corresponding functions in `metatensor::io`.
- `Labels::select` function, to sub-select labels entries.

## [Version 0.1.5](https://github.com/metatensor/metatensor/releases/tag/metatensor-rust-v0.1.5) - 2024-03-12

### Fixed

- Re-export C API bindings in the `metatensor::c_api` module (#549)

## [Version 0.1.4](https://github.com/metatensor/metatensor/releases/tag/metatensor-rust-v0.1.4) - 2024-03-12

### Changed

- Bindings to the raw C API are now part of the `metatensor-sys` crate (#535)

## [Version 0.1.3](https://github.com/metatensor/metatensor/releases/tag/metatensor-rust-v0.1.3) - 2024-02-12

### Fixed

- Fixed build with older versions of rustc by pinning dependencies of
  metatensor-core (#505 and #506)

## [Version 0.1.2](https://github.com/metatensor/metatensor/releases/tag/metatensor-rust-v0.1.2) - 2024-01-26

### Added

- Offer serialization functionality as associated functions (i.e. `TensorMap::load`)
  in addition to the existing free standing functions (i.e. `metatensor::io::load`) (#453)

- Serialization of labels with `Labels::save`, `Labels::load`,
  `Labels::save_buffer`, `Labels::load_buffer` and the corresponding free
  functions. (#455)

## [Version 0.1.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-rust-v0.1.1) - 2024-01-05

### Fixed

- Fixed the build with Cargo 1.75

## [Version 0.1.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-rust-v0.1.0) - 2023-10-11

### Added

- Rust bindings to all metatensor-core class: `Labels`, `LabelsEntry`,
  `TensorBlock`, and `TensorMap`;
- Rust binding of `mts_array_t` through the `Array` trait;
- Implementation of `Array` for [ndarray](https://docs.rs/ndarray/);
- Parallel iteration over labels entries and blocks using [rayon](https://docs.rs/rayon/);
