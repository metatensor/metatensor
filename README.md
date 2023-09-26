# Metatensor

[![tests status](https://img.shields.io/github/checks-status/lab-cosmo/metatensor/master)](https://github.com/lab-cosmo/metatensor/actions?query=branch%3Amaster)
[![documentation](https://img.shields.io/badge/documentation-latest-sucess)](https://lab-cosmo.github.io/metatensor/latest/)

Metatensor is a self-describing sparse tensor data format for atomistic machine
learning and beyond; storing values and gradients of these values together.
Think numpy `ndarray` or pytorch `Tensor` equipped with extra metadata for
atomic systems and other point clouds data. The core of this library is written
in Rust and we provide API for C, C++, and Python.

The main class of metatensor is the `TensorMap` data structure, defining a
custom block-sparse data format. If you are using metatensor from Python, we
additionally provide a collection of mathematical, logical and other utility
operations to make working with TensorMaps more convenient.

## Documentation

For details, tutorials, and examples, please have a look at our [documentation](https://lab-cosmo.github.io/metatensor/latest/).

## Contributors

Thanks goes to all people that make metatensor possible:

[![contributors list](https://contrib.rocks/image?repo=lab-cosmo/metatensor)](https://github.com/lab-cosmo/metatensor/graphs/contributors)

We always welcome new contributors. If you want to help us take a look at our
[contribution guidelines](CONTRIBUTING.rst) and afterwards you may start with an
open issue marked as [good first
issue](https://github.com/lab-cosmo/metatensor/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
