Miscellaneous
=============

Version number
^^^^^^^^^^^^^^

.. doxygenfunction:: metatensor_torch::version

.. c:macro:: METATENSOR_TORCH_VERSION

    Macro containing the compile-time version of metatensor-torch, as a string

.. c:macro:: METATENSOR_TORCH_VERSION_MAJOR

    Macro containing the compile-time **major** version number of
    metatensor-torch, as an integer

.. c:macro:: METATENSOR_TORCH_VERSION_MINOR

    Macro containing the compile-time **minor** version number of
    metatensor-torch, as an integer

.. c:macro:: METATENSOR_TORCH_VERSION_PATCH

    Macro containing the compile-time **patch** version number of
    metatensor-torch, as an integer


Modules containing metatensor data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: metatensor_torch::Module
    :members:


``TensorMap`` Serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: metatensor_torch::save(const std::string &path, TensorMap tensor)

.. doxygenfunction:: metatensor_torch::save_buffer(TensorMap tensor)

.. doxygenfunction:: metatensor_torch::load

.. doxygenfunction:: metatensor_torch::load_buffer



``TensorBlock`` Serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: metatensor_torch::save(const std::string &path, TensorBlock block)

.. doxygenfunction:: metatensor_torch::save_buffer(TensorBlock block)

.. doxygenfunction:: metatensor_torch::load_block

.. doxygenfunction:: metatensor_torch::load_block_buffer


``Labels`` Serialization
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: metatensor_torch::save(const std::string &path, Labels labels)

.. doxygenfunction:: metatensor_torch::save_buffer(Labels labels)

.. doxygenfunction:: metatensor_torch::load_labels

.. doxygenfunction:: metatensor_torch::load_labels_buffer
