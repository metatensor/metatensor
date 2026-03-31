<h1>
<p align="center">
    <img src="https://raw.githubusercontent.com/metatensor/metatensor/refs/heads/main/docs/static/images/metatensor-horizontal-dark.png" alt="Metatensor logo" width="600"/>
</p>
</h1>

<h4 align="center">

[![tests status](https://img.shields.io/github/checks-status/metatensor/metatensor/main)](https://github.com/metatensor/metatensor/actions?query=branch%3Amain)
[![documentation](https://img.shields.io/badge/📚_documentation-latest-sucess)](https://docs.metatensor.org/latest/)
[![coverage](https://codecov.io/gh/metatensor/metatensor/branch/main/graph/badge.svg)](https://codecov.io/gh/metatensor/metatensor)
</h4>

Metatensor is a self-describing sparse tensor data format for atomistic machine
learning and beyond; storing values and gradients of these values together.
Think numpy `ndarray` or pytorch `Tensor` equipped with extra metadata for
atomic systems and other point clouds data. The core of this library is written
in Rust and we provide API for C, C++, and Python.

The main class of metatensor is the `TensorMap` data structure, defining a
custom block-sparse data format. If you are using metatensor from Python, we
additionally provide a collection of mathematical, logical and other utility
operations to make working with TensorMaps more convenient.

If you use metatensor for your work, please read and cite our preprint available on
[arXiv](http://arxiv.org/abs/2508.15704).

# Documentation

For details, tutorials, and examples, please have a look at our [documentation](https://docs.metatensor.org/).

# Contributors

Thanks goes to all people that make metatensor possible:

[![contributors list](https://contrib.rocks/image?repo=metatensor/metatensor)](https://github.com/metatensor/metatensor/graphs/contributors)

We always welcome new contributors. If you want to help us take a look at our
[contribution guidelines](CONTRIBUTING.rst) and afterwards you may start with an
open issue marked as [good first
issue](https://github.com/metatensor/metatensor/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

This project is
[maintained](https://github.com/lab-cosmo/.github/blob/main/Maintainers.md) by
[@Luthaf](https://github.com/Luthaf) and [@HaoZeke](https://github.com/HaoZeke),
who will reply to issues and pull requests opened on this repository as soon as
possible. You can mention them directly if you did not receive an answer after a
couple of days.

<!-- marker-cite -->

# Citing metatensor

If you found metatensor useful for your work, please cite the corresponding article:

F. Bigi, J.W. Abbott, P. Loche et. al.<br>
Metatensor and metatomic: foundational libraries for interoperable atomistic machine learning, (2026).<br>
[https://doi.org/10.1063/5.0304911](https://doi.org/10.1063/5.0304911)

```bibtex
@article{bigi_metatensor_2026,
  title = {Metatensor and Metatomic: {{Foundational}} Libraries for Interoperable Atomistic Machine Learning},
  shorttitle = {Metatensor and Metatomic},
  author = {Bigi, Filippo and Abbott, Joseph W. and Loche, Philip and Mazitov, Arslan and Tisi, Davide and Langer, Marcel F. and Goscinski, Alexander and Pegolo, Paolo and Chong, Sanggyu and Goswami, Rohit and Febrer, Pol and Chorna, Sofiia and Kellner, Matthias and Ceriotti, Michele and Fraux, Guillaume},
  year = 2026,
  month = feb,
  journal = {J. Chem. Phys.},
  volume = {164},
  number = {6},
  pages = {064113},
  issn = {0021-9606},
  doi = {10.1063/5.0304911},
}
```

<!-- marker-end -->
