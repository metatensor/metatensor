Descriptors
===========

.. doxygentypedef:: aml_descriptor_t

The following functions operate on :c:type:`aml_descriptor_t`:

- :c:func:`aml_descriptor`: create new descriptors
- :c:func:`aml_descriptor_free`: free allocated descriptors
- :c:func:`aml_descriptor_sparse_labels`: get the sparse :c:struct:`aml_labels_t` associated with a descriptor
- :c:func:`aml_descriptor_block_by_id`: get a :c:struct:`aml_block_t` in this descriptor from its index
- :c:func:`aml_descriptor_block_selection`: get a :c:struct:`aml_block_t` in this descriptor from its label
- :c:func:`aml_descriptor_sparse_to_features`: move entries from sparse labels to features labels
- :c:func:`aml_descriptor_components_to_features`: move entries from component labels to features labels
- :c:func:`aml_descriptor_sparse_to_samples`: move entries from sparse labels to sample labels

---------------------------------------------------------------------

.. doxygenfunction:: aml_descriptor

.. doxygenfunction:: aml_descriptor_free

.. doxygenfunction:: aml_descriptor_sparse_labels

.. doxygenfunction:: aml_descriptor_block_by_id

.. doxygenfunction:: aml_descriptor_block_selection

.. doxygenfunction:: aml_descriptor_sparse_to_features

.. doxygenfunction:: aml_descriptor_components_to_features

.. doxygenfunction:: aml_descriptor_sparse_to_samples


.. ---------------------------------------------------------------------

.. .. doxygenenum:: rascal_indexes_kind

.. .. doxygenstruct:: rascal_indexes_t
..     :members:

.. .. doxygenstruct:: rascal_densified_position_t
..     :members:
