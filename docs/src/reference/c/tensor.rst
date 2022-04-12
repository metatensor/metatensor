Tensor Maps
===========

.. doxygentypedef:: aml_tensormap_t

The following functions operate on :c:type:`aml_tensormap_t`:

- :c:func:`aml_tensormap`: create new tensor map
- :c:func:`aml_tensormap_free`: free allocated tensor maps
- :c:func:`aml_tensormap_keys`: get the keys defined in a tensor map as :c:struct:`aml_labels_t`
- :c:func:`aml_tensormap_block_by_id`: get a :c:struct:`aml_block_t` in a tensor map from its index
- :c:func:`aml_tensormap_block_selection`: get a :c:struct:`aml_block_t` in a tensor map from its key
- :c:func:`aml_tensormap_keys_to_samples`: move entries from keys to sample labels
- :c:func:`aml_tensormap_keys_to_properties`: move entries from keys to properties labels
- :c:func:`aml_tensormap_components_to_properties`: move entries from component labels to properties labels


---------------------------------------------------------------------

.. doxygenfunction:: aml_tensormap

.. doxygenfunction:: aml_tensormap_free

.. doxygenfunction:: aml_tensormap_keys

.. doxygenfunction:: aml_tensormap_block_by_id

.. doxygenfunction:: aml_tensormap_block_selection

.. doxygenfunction:: aml_tensormap_keys_to_samples

.. doxygenfunction:: aml_tensormap_keys_to_properties

.. doxygenfunction:: aml_tensormap_components_to_properties
