Reduction over samples
======================

These functions allow to reduce over the samples of :py:class:`TensorMap` or
:py:class:`TensorBlock` objects. The values sharing the same values for the
indicated ``sample_names`` will be combined in a single entry. One way to see
these operations is that the sample indices describe the non-zero entries in a
*sparse* array, and the reduction acts much like :func:`numpy.sum`, where
``sample_names`` plays the same role as the ``axis`` argument. Whenever
gradients are present, the reduction is performed also on the gradients.

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
