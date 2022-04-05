Blocks
======

.. doxygentypedef:: aml_block_t

The following functions operate on :c:type:`aml_block_t`:

- :c:func:`aml_block`: create new blocks
- :c:func:`aml_block_free`: free allocated blocks
- :c:func:`aml_block_labels`: get one of the :c:struct:`aml_labels_t` associated with this block
- :c:func:`aml_block_data`: get one of the :c:struct:`aml_array_t` associated with this block
- :c:func:`aml_block_add_gradient`: add gradient data to this block
- :c:func:`aml_block_gradients_list`: get the list of gradients in this block

---------------------------------------------------------------------

.. doxygenfunction:: aml_block

.. doxygenfunction:: aml_block_free

.. doxygenfunction:: aml_block_labels

.. doxygenfunction:: aml_block_data

.. doxygenfunction:: aml_block_add_gradient

.. doxygenfunction:: aml_block_gradients_list

---------------------------------------------------------------------

.. doxygenenum:: aml_label_kind
