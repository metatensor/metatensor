Miscellaneous
=============

Error handling
--------------

.. doxygenclass:: metatensor::Error
    :members:


N-dimensional arrays
--------------------

.. doxygenclass:: metatensor::NDArray
    :members:


Serialization
-------------

.. doxygenfunction:: metatensor::io::save

.. doxygenfunction:: metatensor::io::save_buffer

.. doxygenfunction:: metatensor::io::load

.. doxygenfunction:: metatensor::io::load_buffer(const uint8_t* buffer, size_t buffer_count, mts_create_array_callback_t create_array)

.. doxygenfunction:: metatensor::io::load_buffer(const Buffer& buffer, mts_create_array_callback_t create_array)

.. doxygenfunction:: metatensor::details::default_create_array
