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

### metatensor-core C

#### Changed

- `mts_array_t.move_samples_from` is now `mts_array_t.move_data`, and allows for
  more granular data movement. `mts_sample_mapping_t` has been renamed to
  `mts_data_movement_t`.
- `mts_array_t.data` has been replaced by `mts_array_t.as_dlpack`, returning the
  data using the [dlpack](https://github.com/dmlc/dlpack) standard. TensorBlock can now contain data on different devices and with varied dtypes.
- Serialization of `mts_tensor_t`, `mts_block_t` and `mts_labels_t` can now be
  done even if the data lives on a different device or uses a dtype other than
  float64. `mts_create_array_callback_t` now takes an extra parameter for the
  dtype to use when creating new arrays.
- `mts_tensor_keys_to_samples` now handles merging blocks with different set
  of properties.
- `mts_tensor_keys_to_samples`, `mts_tensor_keys_to_properties` and
  `mts_array_t.create` now take a `fill_value` parameter, which will be used to
  fill missing values when merging blocks or creating new arrays respectively.
- `mts_labels_t` is now an opaque pointer, and stores data using `mts_array_t`;
  allowing to remove the need to copy all data back to CPU when creating labels.
  As part of this, `mts_labels_create` is now `mts_labels`, and
  `mts_labels_create_assume_unique` is now `mts_labels_assume_unique`.
- `mts_last_error` can now return custom data and error origin together with the
  last error message. These can be set by a new function `mts_set_last_error`.

#### Added

- There are new functions `mts_block_device`, `mts_tensormap_device`,
  `mts_block_dtype`, `mts_tensormap_dtype` to access dtype and device of metatensor data.
- There are new functions to work with `mts_labels_t`: `mts_labels_dimensions`,
  `mts_labels_values`, and `mts_labels_values_cpu`.
- `mts_set_last_error` to set the last error that occured in a callback before
  transferring control back to libmetatensor. The data set by this function can
  then be retrieved with `mts_last_error` and used to store and rethrow
  exceptions across the C API boundary.

### metatensor-core C++

#### Changed

- `TensorBlock::values()` is now a template `values<T>()` (defaulting to
  `double`) and returns a `DLPackArray<T>` that owns the DLPack resource,
  preventing dangling-pointer issues.
- `Labels::position`, `Labels::set_intersection`, `Labels::set_difference`, and
  `Labels::select` now returns a `std::optional<size_t>` instead of relying on
  -1 as a sentinel value. The overloads taking `int64_t` pointers are still
  available for a zero overhead version.

#### Added

- `TensorBlock::device`, `TensorMap::device`, `TensorBlock::dtype`, and
  `TensorMap::dtype` functions.
- The `MtsArray` class, which provides a RAII wrapper and more convenient way to
  use `mts_array_t` from C++

### metatensor-core Python

#### Added

- Added `ExternalCudaArray` in Python, the CUDA counterpart to
  `ExternalCpuArray`. It wraps non-Python CUDA data as a `torch.Tensor` via
  DLPack, for use with external array backends that store data on CUDA devices.

#### Removed

- `LabelsView` has been removed, and with it the following functions:
  `Labels.is_view()`, `Labels.to_owned()`, `Labels.view()`, and
  `Labels.__getitem__(list[str])`. We recomend using `Labels.column()` instead to access the values of individual dimensions of Labels.

## [Version 0.1.20](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.1.20) - 2026-02-27

### Fixed

- Ensure the `info` of `TensorMap` is kept in the new tensor when
  `keys_to_samples`, `components_to_properties`, and `key_to_properties` are
  called
- Pin getrandom to make sure the code compiles with rustc 1.74


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
