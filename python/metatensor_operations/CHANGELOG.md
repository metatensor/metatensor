# Changelog

All notable changes to metatensor-operations are documented here, following the
[keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/metatensor/)

<!-- Possible sections

### Added

### Fixed

### Changed

### Removed
-->

## [Version 0.3.2](https://github.com/metatensor/metatensor/releases/tag/metatensor-operations-v0.3.2) - 2025-02-18

### Fixed

- `mean_over_samples` was not computing the correct mean


## [Version 0.3.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-operations-v0.3.1) - 2025-02-04

### Added

- Added a new `filter_blocks` function to remove some blocks from a TensorMap
  according to a selection (#823)

### Fixed

- Fixed the `sort` function when dealing with empty blocks (#840)

## [Version 0.3.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-operations-v0.3.0) - 2024-10-29

### Changed

- `axis="components"` can now be passed to `metatensor.operations.rename_dimension` (#763)

- `split`, `split_block`, `slice` and `slice_block` have been updated to accept
  a list of integer directly specifying which samples/properties to keep in the
  output. The corresponding parameter has been renamed to `selection` or
  `selections` (for `slice` and `split` respectively) (#772)

## [Version 0.2.4](https://github.com/metatensor/metatensor/releases/tag/metatensor-operations-v0.2.4) - 2024-10-11

### Changed

- `block_from_array` now takes optional parameters to specify the names of
  sample, component and property dimensions

## [Version 0.2.3](https://github.com/metatensor/metatensor/releases/tag/metatensor-operations-v0.2.3) - 2024-08-28

### Changed

- We now require Python >= 3.9
- `slice` and `drop_blocks` are now faster thanks to `Labels.select`

## [Version 0.2.2](https://github.com/metatensor/metatensor/releases/tag/metatensor-operations-v0.2.2) - 2024-06-19

### Fixed

- Fixed a bug in `metatensor.torch.sort` where the labels where not properly
  sorted (#647)
- Fix `metatensor.abs` when used with complex values (#553)


## [Version 0.2.1](https://github.com/metatensor/metatensor/releases/tag/metatensor-operations-v0.2.1) - 2024-03-01

### Changed

- Use `torch.jit.script` by default on all operations when using the TorchScript
  backend (i.e. `metatensor.torch`) (#504)

## [Version 0.2.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-operations-v0.2.0) - 2024-02-07

### Changed

- `join()` operation now includes the `sort_samples` argument to control the
  sorting of samples in the output joined TensorMap. This defaults to False, opposite
  to the previous default behaviour.

### Added

- `detach()` operation to detach all values in a TensorMap/TensorBlock from any
  computational graph
- `requires_grad()` operation to set the `requires_grad` of all values in a
  TensorMap/TensorBlock when storing data in torch Tensors.

### Removed

- the `to` operation was removed. Similar functionality is now offered by
  `TensorMap.to`, `TensorBlock.to`, and the operations `detach()` and
  `requires_grad()`.

## [Version 0.1.0](https://github.com/metatensor/metatensor/releases/tag/metatensor-operations-v0.1.0) - 2023-10-11

### Added

- Creation operations: `empty_like()`, `ones_like()`, `zeros_like()`,
  `random_like()`, `block_from_array()`;
- Linear algebra: `dot()`, `lstsq()`, `solve()`;
- Logic function: `allclose()`, `equal()`, `equal_metadata()`;
- Manipulation operations: `drop_blocks()`, `join()`, `manipulate dimension`,
  `one_hot()`, `remove_gradients()`, `samples reduction`, `slice()`, `split()`,
  `to()`;
- Mathematical functions: `abs()`, `add()`, `divide()`, `multiply()`, `pow()`,
  `subtract()`;
- Set operations: `unique_metadata()`, `sort()`;
