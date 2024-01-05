# Changelog

All notable changes to metatensor-torch are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/lab-cosmo/metatensor/)

### metatensor-torch C++

#### Added

- `TensorBlock::to`, `TensorMap::to`, and `System::to` to change the device or
  dtype of torch Tensor stored by metatensor
- `Labels::device`, `TensorBlock::device` and `TensorMap::device`; as well as
  `TensorMap::scalar_type`, and `TensorBlock::scalar_type` to query the current
  device and scalar type/dtype used by the data.

#### Fixed

#### Changed

#### Removed


### metatensor-torch Python

#### Added

- `TensorBlock.to`, `TensorMap.to` and `System.to` to change the device or dtype
  of torch Tensor stored by metatensor
- `Labels.device`, `TensorBlock.device` and `TensorMap.device`; as well as
  `TensorMap.dtype`, and `TensorBlock.dtype` to query the current device and
  dtype used by the data.

#### Fixed

#### Changed

#### Removed


## [Version 0.1.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-torch-v0.1.0) - 2023-10-11

### metatensor-torch C++

#### Added

- TorchScript bindings to all metatensor-core class: `Labels`, `LabelsEntry`,
  `TensorBlock`, and `TensorMap`;
- Implementation of `mts_array_t`/`metatensor::DataArrayBase` for `torch::Tensor`;

### metatensor-torch Python

#### Added

- Expose TorchScript classes to Python;
- Expose all functions from `metatensor-operations` as TorchScript compatible code;
