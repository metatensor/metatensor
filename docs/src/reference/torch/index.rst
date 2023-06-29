.. _torch-api-reference:

TorchScript API reference
=========================

.. py:currentmodule:: equistore.torch

We provide a special PyTorch C++ extension exporting all of the core equistore
types in a way compatible with the TorchScript compiler, allowing users to save
and load models based on equistore everywhere TorchScript is supported. In
particular, this allow to define, train and save a model from Python, and then
load it with pure C++ code, without requiring a Python interpreter. Please refer
to the :ref:`installation instructions <install-torch-script-lib>` to know how
to install the Python and C++ sides of this library.

The classes and functions in the TorchScript API are kept as close as possible
to the classes and functions of the pure Python API, with the explicit goal that
changing from

.. code-block:: python

    import equistore
    from equistore import TensorMap, TensorBlock, Labels

to

.. code-block:: python

    import equistore.torch as equistore
    from equistore.torch import TensorMap, TensorBlock, Labels

should be 80% of the work required to make a model developed in Python with
:py:mod:`equistore.operations` compatible with TorchScript.

The documentation for the three usual core classes of equistore can be found in
the following pages:

.. toctree::
    :maxdepth: 1

    tensor
    block
    labels
    serialization

.. TODO: operations are not yet available

--------------------------------------------------------------------------------

If you want to use equistore's TorchScript API from C++, you might be interested
in the following documentation:

.. toctree::
    :maxdepth: 1

    cxx/index

.. _python-vs-torch:

Differences between Python and TorchScript API
----------------------------------------------

The Python and TorchScript interfaces to equistore are very similar, and have
some level of overlap (it is possible to store data in torch ``Tensor`` with the
Python API), so a big question is when/why should you use one or the other, and
what's the difference between them.

First, the Python and TorchScript API are separate to allow using the Python API
without torch, for example in a pure numpy workflow, or with other array types
such as `Jax`_ arrays, `cupy`_ arrays, *etc.*.

While this works great for a lot of use cases based on torch (defining models,
training them with autograd, …), the TorchScript compiler is more restrictive in
what code it accepts and does not support calling into arbitrary native code
like the Python API does. The TorchScript version of equistore is here to fill
this gap, teaching the TorchScript compiler what's going on inside equistore and
how to translate code using equistore to TorchScript.

Another different is that while the Python API supports storing data in multiple
ways, including storing it on GPU through torch Tensor, the metadata is always
stored on CPU, inside the equistore shared library. The TorchScript API changes
this to enable storing both the data and metadata on GPU, minimizing data
transfer and making the models faster.

If you only care about PyTorch, we would recommend using the TorchScript API
from the start, to make sure you will be able to export your code to
TorchScript. If you are not using PyTorch, or if you want to write code in an
engine agnostic way, we recommend using the Python API.

.. _Jax: https://jax.readthedocs.io/en/latest/
.. _cupy: https://cupy.dev
