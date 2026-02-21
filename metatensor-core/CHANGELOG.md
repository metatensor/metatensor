# Changelog

All notable changes to metatensor-core are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/metatensor/)

<!-- Possible sections for each package:

#### Added

#### Fixed

#### Changed

#### Removed
-->

### Added

- Added `mts_array_t.device` function pointer to query the device of an array
  without exporting via DLPack. Implemented for all array backends (Rust
  `ArcArray`, C++ `SimpleDataArray`/`EmptyDataArray`/`DLPackArray`,
  `TorchDataArray`, and Python numpy/torch arrays).
- C++ `TensorBlock::values()` now accepts optional `device` and `stream`
  parameters, allowing data to be requested on a specific device rather than
  always defaulting to CPU.
- C++ `DLPackArray<T>::device()` accessor returns the DLDevice of the managed
  tensor. `DLPackArray<T>::operator()` now throws for non-CPU data to prevent
  invalid memory access.
- Added `ExternalCudaArray` in Python, the CUDA counterpart to `ExternalCpuArray`.
  It wraps non-Python CUDA data as a `torch.Tensor` via DLPack, for use with
  external array backends (e.g. Rust/Burn) that store data on CUDA devices.
- C++ `TensorBlock::values()` is now a template `values<T>()` (defaulting to
  `double`) and returns a `DLPackArray<T>` that owns the DLPack resource,
  preventing dangling-pointer issues. The data is requested on CPU; if the
  underlying array lives on another device, a copy may occur. For direct GPU
  access without a copy, use the C-level ``as_dlpack`` interface instead.
- Added `mts_tensormap_load_mmap` and `mts_block_load_mmap` C API functions for
  memory-mapped loading of `.mts` files. Data arrays are created via a
  user-provided `mts_create_mmap_array_callback_t`, keeping the core library
  array-agnostic. The callback receives the raw mmap pointer and data shape,
  letting each language binding construct its own array type (e.g.
  `MmapDataArray` in C++, `MmapNdarray` in Rust, `MmapOwner`-backed arrays in
  Python, `torch::from_blob` in PyTorch). Call `mts_mmap_free` to release the
  underlying mapping. Labels are still loaded normally. Corresponding wrappers:
  C++ `metatensor::io::load_mmap`/`load_block_mmap`, `TensorMap::load_mmap`,
  `TensorBlock::load_mmap`; Rust `metatensor::io::load_mmap`/`load_block_mmap`;
  Python `metatensor.load_mmap`/`metatensor.load_block_mmap`.
- Added `MmapDataArray` C++ class (`arrays.hpp`) as a read-only
  `DataArrayBase` backed by memory-mapped data. Supports DLPack export and
  serves as the default C++ mmap array backend.
- Fixed component carry logic in `SimpleDataArray::move_samples_from` for
  arrays with more than one component dimension (4D+ tensors).

### Changed

- `mts_array_t.move_samples_from` is now `mts_array_t.move_data`, and allows for
  more granular data movement. `mts_sample_mapping_t` has been renamed to
  `mts_data_movement_t`.
- `TensorMap::keys_to_samples` now handles merging blocks with different set
  of properties.

### Removed

- Removed `mts_array_t.data` function pointer and all corresponding
  implementations (`DataArrayBase::data()` in C++, `Array::data()` in Rust,
  `_mts_array_data` in Python). Use `mts_array_t.as_dlpack` instead, which
  supports all numeric types via the DLPack standard rather than only float64.
- `LabelsView` has been removed, and with it the following functions:
  `Labels.is_view()`, `Labels.to_owned()`, `Labels.view()`, and
  `Labels.__getitem__(list[str])`. We recomend using `Labels.column()` instead to access the values of individual dimensions of Labels.


## [Version 0.1.20](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.20) - 2026-02-27

### Fixed

- Ensure the `info` of `TensorMap` is kept in the new tensor when
  `keys_to_samples`, `components_to_properties`, and `key_to_properties` are
  called
- Pin getrandom to make sure the code compiles with rustc 1.74


#### Removed

- `LabelsView` has been removed, and with it the following functions:
  `Labels.is_view()`, `Labels.to_owned()`, `Labels.view()`, and
  `Labels.__getitem__(list[str])`. We recomend using `Labels.column()` instead to access the values of individual dimensions of Labels.

## [Version 0.1.19](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.19) - 2025-12-11

### Fixed

- Reset the size of the buffer produced by `mts_tensormap_save_buffer`,
  `mts_block_save_buffer`, and `mts_labels_save_buffer` to the same size as
  v0.1.17

## [Version 0.1.18](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.18) - 2025-12-04

### Added

- It is now possible to store and retrieve global metatadata about a TensorMap,
  in the form of string key/value pairs. The following API are available to
  manipulate this information:

  - in C, you can use `mts_tensormap_set_info`, `mts_tensormap_get_info` and
    `mts_tensormap_info_keys`
  - in C++, you can use `TensorMap::set_info`, `TensorMap::get_info` and
    `TensorMap::info`
  - in Python, you can use `TensorMap.set_info`, `TensorMap.get_info` and
    `TensorMap.info`

### Changed

- We now requires at least cmake v3.22 to compile metatensor
- We now require Python >= 3.10

### Fixed

- `TensorMap::components_to_properties` now properly handles multiple components
  at once.

## [Version 0.1.17](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.17) - 2025-09-02

### Fixed

- The pre-built wheels on PyPI are now compiled in release mode again

### Changed

- We removed some overhead when creating `Labels` by lazily initializing more internal data (#971)

## [Version 0.1.16](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.16) - 2025-08-22

### Fixed

- Fix the use of pre-compiled metatensor from CMake (#954)
- Fix the compilation with rustc >=1.89 and rustc <1.80 (#959)

## [Version 0.1.15](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.15) - 2025-07-29

### Changed

- `TensorMap.to`, `TensorBlock.to`, and `Labels.to` now accept a `non_blocking`
  argument, with the same meaning as in `torch.Tensor.to`.

### Added

- `mts_labels_create_assume_unique` to create `mts_labels` without checking that
  the entries are unique; as well as the corresponding flag in the C++ and
  Python constructors. This allows bypassing a check when the user can ensure
  beforehand that all entries will be unique.

## [Version 0.1.14](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.14) - 2025-04-11

### Fixed

- The `metatensor::shared` and `metatensor::static` targets in CMake are no longer
  global, allowing multiple calls to `find_package(metatensor)`

## [Version 0.1.13](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.13) - 2025-04-10

### metatensor-core C

#### Added

- `mts_labels_difference` for finding the set difference between two `mts_labels_t`

### metatensor-core C++

#### Added

- `Labels::set_difference` for finding the set difference between two `Labels`

### metatensor-core Python

#### Added

- `Labels.difference` and `Labels.difference_and_mapping` for finding the set
  difference between two `Labels`

## [Version 0.1.12](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.12) - 2025-02-17

### Changed

- The default extension for file serialization is now `.mts` instead of `.npz`

## [Version 0.1.11](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.11) - 2024-10-23

### Changed

- The code now requires Rustc v1.74 to build.
- Labels creation is quite a bit faster for large labels thanks to #752 and #754

## [Version 0.1.10](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.10) - 2024-08-28

### metatensor-core C++

#### Added

- `Labels::select` to sub-select entries in labels
- Added support for serialization of TensorBlock with `TensorBlock::load`,
  `TensorBlock::load_buffer`, `TensorBlock::save`, `TensorBlock::save_buffer`
  and the corresponding functions in `metatensor::io`.

### metatensor-core C

#### Added

- `mts_labels_select` to sub-select entries in labels
- Added support for serialization of TensorBlock with `mts_block_load`,
  `mts_block_load_buffer`, `mts_block_save`, and `mts_block_save_buffer`.


### metatensor-core Python

#### Added

- `Labels.select` to sub-select entries in labels
- Added support for serialization of TensorBlock with `TensorBlock.load`,
  `TensorBlock.load_buffer`, `TensorBlock.save`, `TensorBlock.save_buffer`
  and the corresponding functions in `metatensor.io`.

## [Version 0.1.9](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.9) - 2024-07-15

### metatensor-core Python

#### Added

- `TensorBlock.__len__` and `TensorBlock.shape`, which return the length and
  shape of the values in the block respectively
- We can now load (but not save) TensorMap stored in npz files using DEFLATE
  compression (#671)

#### Changed

- We now require Python >= 3.9

#### Fixed

- Fixed a memory leak affecting all data stored in TensorBlock (#683)

## [Version 0.1.8](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.8) - 2024-05-13

### Fixed

- fix the build when using metatensor from the Rust bindings

## [Version 0.1.7](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.7) - 2024-05-13

### metatensor-core C/C++

#### Added

- preprocessor macros containing the version number of metatensor:
  `METATENSOR_VERSION`, `METATENSOR_VERSION_MAJOR`, `METATENSOR_VERSION_MINOR`,
  and `METATENSOR_VERSION_PATCH`.

#### Changed

- installation configuration in CMake now uses the standard `GNUInstallDirs`

#### Fixed

- removed dependency on `bcrypt` on Windows
- the shared libraries is installed with execute permissions

## [Version 0.1.6](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.6) - 2024-05-03

### metatensor-core C/C++

#### Fixed

- fixed compilation with Rust 1.78 (#605)
- fixed compilation on some Windows systems (#575)

## [Version 0.1.5](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.5) - 2024-04-09

### metatensor-core C++

#### Fixed

- fixed compilation with cmake 3.29.1 (#573)

### metatensor-core Python

### Changed

-  allow positional arguments in `TensorMap.to`/`TensorBlock.to` (#556)


## [Version 0.1.4](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.4) - 2024-03-01

### Fixed

- fixed compilation on macOS (#525)
- added checks for the size of the new values in `Labels.insert` (#519)

## [Version 0.1.3](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.3) - 2024-02-12

### Fixed

- Fixed the build with Cargo 1.65 (#505)
- Pin dependencies for metatensor-core, ensuring reproducibility of the build as
  new dependencies versions are published (#506)

## [Version 0.1.2](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.2) - 2024-01-26

### metatensor-core C++

#### Added

- Offer serialization functionality as free standing functions (i.e. `metatensor::io::load`)
  in addition to the existing associated functions (i.e. `TensorMap::load`) (#453)
- Serialization of labels with `Labels::save`, `Labels::load`,
  `Labels::save_buffer`, `Labels::load_buffer` and the corresponding free
  functions. (#455)

#### Changed

- `TensorMap::save_string_buffer` has been replaced with the template version of
  `TensorMap::save_buffer`


### metatensor-core C

#### Added

- `mts_labels_save`, `mts_labels_load`, `mts_labels_save_buffer`, and
  `mts_labels_load_buffer` to handle serialization of `mts_labels_t`


### metatensor-core Python

#### Added

- `TensorMap.to` and `TensorBlock.to` to change the device, dtype, or backend
  (numpy or torch) of all arrays stored by metatensor
- `Labels.device`, `TensorBlock.device` and `TensorMap.device`; as well as
  `TensorMap.dtype`, and `TensorBlock.dtype` to query the current device and
  dtype used by the data.
- Offer serialization functionality as member functions (i.e. `TensorMap.load`)
  in addition to the existing free standing functions (i.e. `metatensor.load`) (#453)
- Serialization of labels with `Labels.save`, `Labels.load`,
  `Labels.save_buffer`, `Labels.load_buffer` and the corresponding free
  functions.

## [Version 0.1.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.1) - 2024-01-05

### Fixed

- Fixed the build with Cargo 1.75 (#438)
- Allowed saving and loading empty TensorMap; i.e. TensorMap where one of the
  dimensions of the array has 0 elements (#419)

## [Version 0.1.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.0) - 2023-10-11

### metatensor-core C

#### Added

- Initial implementation of all the core classes of metatensor: `mts_labels_t`,
  `mts_block_t`, `mts_tensormap_t`, `mts_array_t` and the corresponding
  functions;
- Serialization for `mts_tensormap_t` using a format derived from [numpy's
  npz](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) format

### metatensor-core C++

#### Added

- C++ API for all the C data structures as C++ classes: `Labels`, `TensorBlock`,
  `TensorMap`;
- Wrapper around `mts_array_t` as an abstract base class `DataArrayBase`;
- Basic implementations of `DataArrayBase` in `SimpleDataArray` and
  `EmptyDataArray`;
- Basic n-dimensional array class `NDArray<T>`, intended to give a minimal API
  to use data stored in `TensorBlock` even if this data does not come from C++
  initially;

### metatensor-core Python

#### Added

- Python API for all the C data structures as Python classes: `Labels`,
  `LabelsEntry`, `TensorBlock`, `TensorMap`;
- Wrapper around `mts_array_t` as an abstract base class `metatensor.data.Array`;
- Implementation of `metatensor.data.Array` with `numpy.ndarray` and
  `torch.Tensor`;
