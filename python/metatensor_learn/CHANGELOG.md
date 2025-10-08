# Changelog

All notable changes to metatensor-learn are documented here, following the
[keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/metatensor/)

<!-- Possible sections

### Added

### Fixed

### Changed

### Removed
-->

### Changed

- We now require Python >= 3.10

### Added

- A custom class `metatensor.learn.nn.Module` that should be used instead of
  `torch.nn.Module` when the modules contains metatensor data (Labels,
  TensorBlock, TensorMap) as attributes. This class will properly handle moving
  this data to the correct dtype and device when calling `module.to()` and
  related functions. It will also handle putting this data in the module
  `state_dict()` and loading it back with `load_state_dict()`.

## [Version 0.3.2](https://github.com/metatensor/metatensor/releases/tag/metatensor-learn-v0.3.2) - 2025-04-25

- Make the code compatible with metatensor-torch v0.7.6

## [Version 0.3.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-learn-v0.3.1) - 2025-02-03

### Fixed

- Indexing inside a `Dataset` is now O(1) instead of O(N) (#790)
- Fixed a bug with the default `invariant_keys` in `metatensor.learn.nn` modules (#785)


## [Version 0.3.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-learn-v0.3.0) - 2024-10-30

### Added

- Added `metatensor.learn.nn.EquivariantTransformation` to apply any
  `torch.nn.Module` to invariants computed from the norm over components of covariant
  blocks. The transformed invariants are then elementwise multiplied back to the
  covariant blocks. For invariant blocks, the `torch.nn.Module` is applied as is (#744)

### Changed

- `metatensor.learn.nn` modules `InvariantTanh`, `InvariantSiLU`, `InvariantReLU`,
  `InvariantLayerNorm`, and `EquivariantLinear` have removed and replaced parameter.
  `invariant_key_idxs` is replaced by `invariant_keys`, a `Labels` object that selects
  for invariant blocks.
- `metatensor.learn.nn` modules `LayerNorm`, `InvariantLayerNorm`, `Linear`, and
  `EquivariantLinear` have altered accepted types for certain parameters. Parameters
  `eps`, `elementwise_affine`, `bias`, and `mean` for the layer norm modules, and `bias`
   for the linear modules are affected. Previously these could be passed as list, but
   now can only be passed as a single value. For greater control over modules applied to
   individual blocks, users are encouraged to use the `ModuleMap` module from
   `metatensor.learn.nn`.

## [Version 0.2.3](https://github.com/metatensor/metatensor/releases/tag/metatensor-learn-v0.2.3) - 2024-08-28

### Changed

- We now require Python >= 3.9
- Dataset and DataLoader can now handle fields with a name which is not a valid
  Python identifier.

## [Version 0.2.2](https://github.com/metatensor/metatensor/releases/tag/metatensor-learn-v0.2.2) - 2024-05-16

### Added

- Added torch-style activation function module maps to `metatensor.learn.nn`: `ReLU`,
  `InvariantReLU`, `SiLU`, and `InvariantSiLU` (#597)
- Added torch-style neural network module maps to `metatensor.learn.nn`:
  `LayerNorm`, `InvariantLayerNorm`, `EquivariantLinear`, `Sequential`, `Tanh`,
  and `InvariantTanh` (#513)

### Fixed

- `metatensor.learn.nn` modules `LayerNorm` and `InvariantLayerNorm` now applies
  sample-independent transformations to input tensors.
- Set correct device for output of when torch default device is different than input device (#595)

## [Version 0.2.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-learn-v0.2.1) - 2024-03-01

### Changed

- `metatensor-learn` is no longer re-exported from `metatensor` and
  `metatensor.torch`, all functions are still available inside
  `metatensor.learn` and `metatensor.torch.learn`.

### Fixed

- Make sure the `Dataset` class is iterable (#500)

## [Version 0.2.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-learn-v0.2.0) - 2024-02-07

### Changed

- Pluralization removed for special kwarg `sample_ids` of `IndexedDataset` ->
  `sample_id`, and provided collate functions `group` and `group_and_join`
  updated accordingly.

### Fixed

- Removal of usage of Labels.range in nn modules to support torch.jit.save (#410)

## [Version 0.1.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-learn-v0.1.0) - 2024-01-26

### Added

- `ModuleMap` and `Linear` modules, following torch.nn.ModuleDict and
  torch.nn.Linear in PyTorch but adapted for `TensorMap`'s (#427)
- `Dataset` and `DataLoader` facilities, following the corresponding classes in
  PyTorch (#428)
