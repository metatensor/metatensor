Reduction over samples
======================

These functions allow to reduce over the sample indices of a
:py:class:`TensorMap` or :py:class:`TensorBlock` objects, generating a new
object. The values sharing the same indices for the indicated ``sample_names``
will be combined in a single entry. The functions differ by the type of
reduction operation, but otherwise operate in the same way. The reduction
operation loops over the samples in each block/map, and combines all those that
only differ by the values of the indices associated with the names listed in the
``sample_names`` argument. One way to see these operations is that the sample
indices describe the non-zero entries in a *sparse* array, and the reduction
acts much like :func:`numpy.sum`, where ``sample_names`` plays the same role as
the ``axis`` argument. Whenever gradients are present, the reduction is
performed also on the gradients.

See also :py:func:`metatensor.sum_over_samples_block` and
:py:func:`metatensor.sum_over_samples` for a detailed discussion with examples.

TensorMap operations
--------------------

.. autofunction:: metatensor.sum_over_samples

.. autofunction:: metatensor.mean_over_samples

.. autofunction:: metatensor.var_over_samples

.. autofunction:: metatensor.std_over_samples

TensorBlock operations
----------------------

.. autofunction:: metatensor.sum_over_samples_block

.. autofunction:: metatensor.mean_over_samples_block

.. autofunction:: metatensor.var_over_samples_block

.. autofunction:: metatensor.std_over_samples_block
