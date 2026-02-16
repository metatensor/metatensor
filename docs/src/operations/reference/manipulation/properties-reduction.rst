Reduction over properties
=========================

These functions allow to reduce over the properties of :py:class:`TensorMap` or
:py:class:`TensorBlock` objects. The values sharing the same values for the
indicated ``property_names`` will be combined in a single entry. One way to see
these operations is that the property indices describe the non-zero entries in a
*sparse* array, and the reduction acts much like :func:`numpy.sum`, where
``property_names`` plays the same role as the ``axis`` argument. Whenever
gradients are present, the reduction is performed also on the gradients.

TensorMap operations
--------------------

.. autofunction:: metatensor.sum_over_properties

.. autofunction:: metatensor.mean_over_properties

.. autofunction:: metatensor.var_over_properties

.. autofunction:: metatensor.std_over_properties

TensorBlock operations
----------------------

.. autofunction:: metatensor.sum_over_properties_block

.. autofunction:: metatensor.mean_over_properties_block

.. autofunction:: metatensor.var_over_properties_block

.. autofunction:: metatensor.std_over_properties_block
