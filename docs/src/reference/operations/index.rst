.. _python-api-operations:

Python operations reference
===========================

The Python API for equistore also provides functions which operate on
:py:class:`equistore.TensorMap`, :py:class:`equistore.TensorBlock`, and
:py:class:`equistore.Labels` and can be used to build your own Machine Learning
models.

The list of all operations currently implemented is available below. If you need
any other operation, please `open an issue
<https://github.com/lab-cosmo/equistore/issues>`_!

Using operations with PyTorch
-----------------------------

`PyTorch`_ is a very popular framework for machine learning, providing multiple
tools to make writing and training models easier. There are two ways to use
the operations with PyTorch:

- Using the pure Python version of equistore, one can store values in a
  :py:class:`equistore.TensorBlock` using :py:class:`torch.Tensor`. In this
  case, all operations will be compatible with torch autograd (automatic
  gradient tracking and differentiation). This allows to train models from
  Python, but not to export the models to run without the Python interpreter.
  When running a model with the pure Python version of equistore, you should use
  the operations from ``equistore.<operation_name>``.

- When using the :ref:`TorchScript version of equistore <torch-api-reference>`,
  one can also compile the Python code to TorchScript and then run the model
  without a Python interpreter. This is particularly useful to export and then
  use an already trained model, for example to run molecular simulations. If you
  want to do this, you should use classes and operations from
  ``equistore.torch``, i.e. :py:class:`equistore.torch.TensorMap` and using the
  operation from ``equistore.torch.<operation_name>``. All the operation above
  are available in the ``equistore.torch`` module, but some might not yet be
  fully TorchScript compatible.

.. _PyTorch: https://pytorch.org/

List of all operations
----------------------

.. toctree::
    :maxdepth: 2

    checks/index
    creation/index
    linear_algebra/index
    logic/index
    manipulation/index
    math/index
    set/index
