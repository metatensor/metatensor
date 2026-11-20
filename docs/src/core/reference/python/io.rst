.. _python-api-io:

Serialization
=============

.. autofunction:: metatensor.save

.. autofunction:: metatensor.io.save_buffer

Loading :py:class:`TensorMap`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: metatensor.load

.. autofunction:: metatensor.io.load_buffer

Loading :py:class:`TensorBlock`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: metatensor.load_block

.. autofunction:: metatensor.io.load_block_buffer

Loading :py:class:`Labels`
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: metatensor.load_labels

.. autofunction:: metatensor.io.load_labels_buffer

Loading with custom arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autodata:: metatensor.io.CreateArrayCallback

    Callback type for creating arrays when loading data. The arguments are a
    pointer to the shape, the number of elements in the shape, a ``DLDataType``
    describing the data type of the array, and a pointer to the ``mts_array_t``
    to be filled. The callback should return ``None``.

.. autofunction:: metatensor.io.load_custom_array

.. autofunction:: metatensor.io.load_buffer_custom_array

.. autofunction:: metatensor.io.load_block_custom_array

.. autofunction:: metatensor.io.load_block_buffer_custom_array

.. autofunction:: metatensor.io.load_labels_custom_array

.. autofunction:: metatensor.io.load_labels_buffer_custom_array

.. autofunction:: metatensor.io.create_numpy_array()

.. autofunction:: metatensor.io.create_torch_array()
