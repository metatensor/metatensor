# Changelog

All notable changes to metatensor-torch are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/metatensor/)

<!-- Possible sections for each package:

### Added

### Fixed

### Changed

### Removed
-->

### Added
- Added support for PyTorch v2.10

## [Version 0.8.3](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.8.1) - 2025-12-05

### Added

- It is now possible to store and retrieve global metatadata about a TensorMap,
  in the form of string key/value pairs. You can use `TensorMap::set_info`,
  `TensorMap::get_info` and `TensorMap::info` to manipulate this information.

## [Version 0.8.2](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.8.1) - 2025-11-05

### Changed

- We now requires at least cmake v3.22 to compile metatensor-torch
- We now require Python >= 3.10

### Added

- Added support for PyTorch v2.9

## [Version 0.8.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.8.1) - 2025-09-24

### Fixed

- The custom `metatensor_torch::Module` now properly handles nested containers
  (list, dict, tuples) with empty containers inside (#978)

## [Version 0.8.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.8.0) - 2025-08-22

### Removed

- The `metatensor.torch.atomistic` python module now lives in the
  `metatomic-torch` package (https://github.com/metatensor/metatomic); as well
  as the corresponding C++ code (`#include <metatensor/torch/atomistic.hpp>`).

### Added

- Added support for PyTorch v2.8
- A custom C++ class `metatensor_torch::Module` that should be used instead of
  `torch::jit::Module` when the modules contains metatensor data (Labels,
  TensorBlock, TensorMap) as attributes. This class will properly handle moving
  this data to the correct dtype and device when calling `module.to()`. The
  corresponding Python class is defined in `metatensor-learn`.
- There is now an `assume_unique` flag in the `Labels` constructor. This allows
  bypassing a check when the user can ensure beforehand that all entries will be
  unique.

### Changed

- `TensorMap.to`, `TensorBlock.to`, and `Labels.to` now accept a `non_blocking`
  argument, with the same meaning as in `torch.Tensor.to`.
- `metatensor.torch.save`, `metatensor.torch.load`,
  `metatensor.torch.load_block` and `metatensor.torch.load_labels` can now take
  `pathlib.Path` objects as inputs, as well as pre-opened files. When using these
  functions from TorchScript, only `str` arguments are supported.

## [Version 0.7.6](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.7.6) - 2025-04-25

### Added

- Added support for PyTorch v2.7

### Changed

- The lowest supported version of PyTorch was increased from v1.12 to v2.1

## [Version 0.7.5](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.7.5) - 2025-04-10

### Added

- `Labels.difference` and `Labels.difference_and_mapping` for finding the set
  difference between two `Labels`

- `energy_uncertainty`, `non_conservative_forces` and `non_conservative_stress` are now
  possible outputs for atomistic models

- The ASE calculator can now use `non_conservative_forces` and `non_conservative_stress`
  outputs, which are selected with the `non_conservative` flag

## [Version 0.7.4](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.7.4) - 2025-03-28

### Added

- Added `MetatensorCalculator.compute_energy()` to provide batched calculation of
  energies, forces and stresses (#880).

### Changed

- `MetatensorCalculator.run_model()` now supports batching across multiple
  `ase.Atoms` objects (#879).

## [Version 0.7.3](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.7.3) - 2025-02-19

### Changed

- The `MetatensorCalculator` for ASE now only computes the stress if the system is
  periodic in all directions


## [Version 0.7.2](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.7.2) - 2025-02-18

### Fixed

- Fixed a bug with the custom TorchScript extensions when using `featomic-torch`

## [Version 0.7.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.7.1) - 2025-02-17

### Changed

- The default extension for file serialization is now `.mts` instead of `.npz`

### Added

- Support for Python 3.13 and PyTorch 2.6


## [Version 0.7.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.7.0) - 2025-01-22

### Changed

- Custom data stored in `System` is now stored as `TensorMap` instead of
  `TensorBlock`. This allow for more kinds of data to be stored.

- `metatensor_torch::TorchLabels` is now `metatensor_torch::Labels`; and
  similarly for `metatensor_torch::TensorBlock` and similarly for
  `metatensor_torch::TensorMap`. We temporarily provide type aliases for
  backward compatibilty.

### Added

- `System` can now be serialized and saved to a file using
  `metatensor.torch.atomistic.save` and
  `metatensor.torch.atomistic.load_system`. For the initial implementation,
  these functions are only accessible from Python.

## [Version 0.6.3](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.6.3) - 2024-12-11

### Fixed

- Fixed an issue with Python wheels building, where the underlying C++ library
  would get a different version from the Python library and fail to load.

## [Version 0.6.2](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.6.2) - 2024-12-11

### Fixed

- Selecting from empty `Labels` no longer fails (#792)
- Models are exported with `requires_grad=False` for all parameters, which
  should make them faster when using autograd to compute forces and stress (#802)

## [Version 0.6.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.6.1) - 2024-11-7

### Fixed

- Added missing data to `NeighborListOptions` serialization (#784)

### Changed

- `load_atomistic_model` now returns a `MetatensorAtomisticMode` instead of a
  raw TorchScript model. This allows to reload, modify and re-export a model
  (#783).

## [Version 0.6.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.6.0) - 2024-10-29

### Added

- a `is_atomistic_model` (Python only) function to check if an loaded module is an
  metatensor atomistic model (#697, #766)

### Changed

- the `System` class now supports boundary conditions along some axes but not
  others. This is implemented via a new `pbc` attribute. Any non-periodic
  dimension in a `System` must have the corresponding cell vector set to zero.
- `NeighborListOptions` can now request `strict` or non-strict neighbor lists (#770)

## [Version 0.5.5](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.5.5) - 2024-09-03

### Added

- a `"features"` standard output for atomistic models (#718)

### Fixed

- the Python wheels request the right versions of torch in their metadata (#724)

## [Version 0.5.4](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.5.4) - 2024-08-28

### Added

- `read_model_metadata` to load only the `ModelMetadata` from an exported
  atomistic model without having to load the whole model.
- Users can now store arbitrary additional metadata in `ModelMetadata.extra`
- Added `Labels.select` function to sub-select entries in labels
- Added support for serialization of TensorBlock with `TensorBlock::load`,
  `TensorBlock::load_buffer`, `TensorBlock::save`, `TensorBlock::save_buffer`
  and the corresponding functions in `metatensor.torch`.

## [Version 0.5.3](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.5.3) - 2024-07-15

### Changed

- `MetatensorAtomisticModel.save()` always saves models on the CPU.
- We now require Python >= 3.9

### Fixed

- Fixed a memory leak in `register_autograd_neighbors` (#684)

## [Version 0.5.2](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.5.2) - 2024-06-21

### Added

- `MetatensorAtomisticModel.save()` to save a wrapped model to a file.
- `TensorBlock.__len__` and `TensorBlock.shape`, which return the length and
  shape of the values in the block respectively (#640)
- `metatensor.torch.atomistic.ase_calculator.MetatensorCalculator` can now use
  [`vesin`](https://github.com/Luthaf/vesin) for faster neighbor list
  calculations (#659)
- When running atomistic models in the PyTorch profiler, different sections of
  the code now have meaningful names

### Deprecated

- `MetatensorAtomisticModel.export()` is deprecated in favor of `MetatensorAtomisticModel.save()`

### Fixed

- `metatensor.torch.atomistic.ase_calculator.MetatensorCalculator` uses the
  right device when computing stress/virial (#660)

## [Version 0.5.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.5.1) - 2024-05-14

### Added

- preprocessor macros containing the version number of metatensor-torch:
  `METATENSOR_TORCH_VERSION`, `METATENSOR_TORCH_VERSION_MAJOR`,
  `METATENSOR_TORCH_VERSION_MINOR`, and `METATENSOR_TORCH_VERSION_PATCH`.

## [Version 0.5.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.5.0) - 2024-05-02

### Changed

- We renamed `neighbors_list` to `neighbor_list` in all functions (#587)

### metatensor-torch Python

#### Changed

- The neighbor lists calculation in `MetatensorCalculator` (ASE calculator based
  on metatensor models) is now a lot faster (#586)
- Multiple small improvements related to custom TorchScript extensions (#584)
- There are reference output for neighbor list calculations, which should help
  writing interfaces to metatensor models in new simulation engines (#588)
- The wheels for `metatensor-torch` on PyPI now declare which versions of torch
  they are compatible with (#592)

## [Version 0.4.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.4.0) - 2024-04-11

### metatensor-torch C++

#### Added

- `ModelCapabilities::dtype`, used by the model to communicate the dtype it
  wants to use for inputs and outputs.
- The `load_model_extensions()` function to facilitate loading a model using
  TorchScript extensions.

#### Changed

- `System::add_data` now has an `override` parameter to replace custom data with
  a new value.

### metatensor-torch Python

#### Changed

- We now release wheels compatible with multiple torch versions on PyPI,
  removing the need to compile C++ code when installing this package.

#### Added

- `ModelCapabilities.dtype`, used by the model to communicate the dtype it
  wants to use for inputs and outputs.
- The `device` that should be used by a model inside the ASE's
  `MetatensorCalculator` can now be specified by the user.
- The `load_model_extensions()` and `load_atomistic_model` functions to
  facilitate loading a model using TorchScript extensions

## [Version 0.3.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.3.0) - 2024-03-01

### metatensor-torch C++

#### Added

- `ModelMetadata` to record metadata about a specific model such as it's name,
  authors, etc.
- Added `interaction_range` and `supported_devices` to `ModelCapabilities`

#### Changed

- `System::species` has been renamed to `System::types`.

### metatensor-torch Python

#### Added

- `ModelMetadata` to record metadata about a specific model such as it's name,
  authors, etc.
- Added `interaction_range` and `supported_devices` to `ModelCapabilities`

#### Changed

- `System.species` has been renamed to `System.types`.

## [Version 0.2.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.2.1) - 2024-01-26

### metatensor-torch C++

#### Added

- Offer serialization functionality as member functions (i.e. `TensorMap::load`)
  in addition to the existing free standing functions (i.e. `metatensor_torch::load`) (#453)
- In-memory serialization with `TensorMap::save_buffer`, `TensorMap::load_buffer`,
  and the respective free standing functions (#455)
- Serialization of Labels, with the same API as `TensorMap` (#455)


### metatensor-torch Python

#### Added

- Offer serialization functionality as member functions (i.e. `TensorMap.load`)
  in addition to the existing free standing functions (i.e. `metatensor.torch.load`) (#453)
- In-memory serialization with `TensorMap.save_buffer`, `TensorMap.load_buffer`,
  and the respective free standing functions (#455)
- Serialization of Labels, with the same API as `TensorMap` (#455)

## [Version 0.2.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.2.0) - 2024-01-08

### metatensor-torch C++

#### Added

- New classes specifically tailored for atomistic models (#405):
  - `System` defines the input of a model;
  - `NeighborsListOptions` allow a model to request a specific neighbors list;
  - `ModelRunOptions`, `ModelOutput` and `ModelCapabilities` allow to statically
    describe capabilities of a model, and request specific outputs from it.

- `TensorBlock::to`, `TensorMap::to`, and `System::to` to change the device or
  dtype of torch Tensor stored by metatensor
- `Labels::device`, `TensorBlock::device` and `TensorMap::device`; as well as
  `TensorMap::scalar_type`, and `TensorBlock::scalar_type` to query the current
  device and scalar type/dtype used by the data.
- `metatensor_torch::version` function, returning the version of the code as a
  string.

#### Fixed

- We now check that all tensors in a `TensorBlock`/`TensorMap` have the same
  dtype and device (#414)
- `keys_to_properties`, `keys_to_samples` and `components_to_properties` now
  keep the different Labels on the same device (#411)

### metatensor-torch Python

#### Added

- New classes specifically tailored for atomistic models (#405):
  - same classes as the C++ interfaces, in `metatensor.torch.atomistic`
  - `MetatensorAtomisticModel` as a way to wrap user-defined `torch.nn.Module`
    and export them in a unified way, handling unit conversions and metadata
    checks.
- ASE calculator based on `MetatensorAtomisticModel` in
  `metatensor.torch.atomistic.ase_calculator`. This allow using arbitrary
  user-defined models to run simulations with ASE.

- `TensorBlock.to`, `TensorMap.to` and `System.to` to change the device or dtype
  of torch Tensor stored by metatensor
- `Labels.device`, `TensorBlock.device` and `TensorMap.device`; as well as
  `TensorMap.dtype`, and `TensorBlock.dtype` to query the current device and
  dtype used by the data.


## [Version 0.1.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-torch-v0.1.0) - 2023-10-11

### metatensor-torch C++

#### Added

- TorchScript bindings to all metatensor-core class: `Labels`, `LabelsEntry`,
  `TensorBlock`, and `TensorMap`;
- Implementation of `mts_array_t`/`metatensor::DataArrayBase` for `torch::Tensor`;

### metatensor-torch Python

#### Added

- Expose TorchScript classes to Python;
- Expose all functions from `metatensor-operations` as TorchScript compatible code;
