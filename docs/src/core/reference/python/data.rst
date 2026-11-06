.. _python-api-array:

Data arrays
===========

This part of metatensor contains advanced functionalities related to how data is
stored inside :py:class:`metatensor.TensorBlock`. Most users should not have to
interact with this!

.. autoclass:: metatensor.Array()

.. _python-api-array-backend:

Data allocated outside of Python
--------------------------------

Data stored in a :py:class:`metatensor.TensorMap`,
:py:class:`metatensor.TensorBlock` or :py:class:`metatensor.Labels` can be
allocated outside of Python, for example by the metatensor library itself, or by
code written in another language (C, C++, Rust, ...) and given to Python with
:py:meth:`metatensor.TensorMap.unsafe_from_ptr` and related functions.

Data allocated by Python (i.e. :py:class:`numpy.ndarray` or
:py:class:`torch.Tensor` given to metatensor) is always returned as-is. Other
data is accessed through `DLPack`_, without copying it, and returned using one
of the following array backends:

- ``"numpy"``: data is returned as :py:class:`numpy.ndarray`. This requires the
  data to live on CPU;
- ``"torch"``: data is returned as :py:class:`torch.Tensor`.

The array backend is chosen with the ``array_backend`` parameter of
:py:meth:`metatensor.TensorMap.unsafe_from_ptr`,
:py:meth:`metatensor.TensorBlock.unsafe_from_ptr`,
:py:meth:`metatensor.Labels.unsafe_from_ptr`,
:py:func:`metatensor.io.load_custom_array` and similar loading functions. It is
then used by all the objects created from these, (blocks, gradients, labels, the
output of operations like :py:meth:`metatensor.Labels.union`, ...).

The ``dtype`` and ``device`` of blocks and labels are returned using the types
of the corresponding array backend (e.g. :py:class:`torch.dtype` and
:py:class:`torch.device` for ``"torch"``).

To use the data with other array libraries,
:py:meth:`metatensor.TensorBlock.values_dlpack` and
:py:meth:`metatensor.Labels.values_dlpack` give access to the values of blocks
and labels as :py:class:`ctypes_dlpack.DLPackArray`, regardless of where the
data was allocated. These can be given to any library supporting DLPack, for
example ``jax.dlpack.from_dlpack(array)``.

.. _DLPack: https://dmlc.github.io/dlpack/latest/

Deprecated functionalities
--------------------------

.. autofunction:: metatensor.register_external_data_wrapper

.. autoclass:: metatensor.ExternalCpuArray

.. autoclass:: metatensor.ExternalCudaArray
