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


``TensorMap`` Serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: metatensor_torch::save(const std::string &path, TorchTensorMap tensor)

.. doxygenfunction:: metatensor_torch::save_buffer(TorchTensorMap tensor)

.. doxygenfunction:: metatensor_torch::load

.. doxygenfunction:: metatensor_torch::load_buffer



``TensorBlock`` Serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: metatensor_torch::save(const std::string &path, TorchTensorBlock block)

.. doxygenfunction:: metatensor_torch::save_buffer(TorchTensorBlock block)

.. doxygenfunction:: metatensor_torch::load_block

.. doxygenfunction:: metatensor_torch::load_block_buffer


``Labels`` Serialization
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: metatensor_torch::save(const std::string &path, TorchLabels labels)

.. doxygenfunction:: metatensor_torch::save_buffer(TorchLabels labels)

.. doxygenfunction:: metatensor_torch::load_labels

.. doxygenfunction:: metatensor_torch::load_labels_buffer
