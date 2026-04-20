.. _python-api-array:

Data arrays
===========

This part of metatensor contains advanced functionalities related to how data is
stored inside :py:class:`metatensor.TensorBlock`. Most users should not have to
interact with this!

.. autoclass:: metatensor.Array()

--------------------------------------------------------------------------------

.. autofunction:: metatensor.register_external_data_wrapper

.. autoclass:: metatensor.ExternalCpuArray

.. autoclass:: metatensor.ExternalCudaArray
