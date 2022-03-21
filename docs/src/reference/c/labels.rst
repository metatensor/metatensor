Labels
======

.. doxygenstruct:: aml_labels_t
    :members:

.. The following functions operate on :c:type:`rascal_descriptor_t`:

.. - :c:func:`rascal_descriptor`: create new descriptors
.. - :c:func:`rascal_descriptor_free`: free allocated descriptors
.. - :c:func:`rascal_calculator_compute`: run the actual calculation
.. - :c:func:`rascal_descriptor_values`: get the values out of the descriptor
.. - :c:func:`rascal_descriptor_gradients`: get the gradients out of the descriptor
.. - :c:func:`rascal_descriptor_indexes`: get the values of one of the indexes of the descriptor
.. - :c:func:`rascal_descriptor_densify`: move some indexes variables from samples to features
.. - :c:func:`rascal_descriptor_densify_values`: advanced version of ``rascal_descriptor_densify``

.. ---------------------------------------------------------------------

.. .. doxygenfunction:: rascal_descriptor

.. .. doxygenfunction:: rascal_descriptor_free

.. .. doxygenfunction:: rascal_descriptor_values

.. .. doxygenfunction:: rascal_descriptor_gradients

.. .. doxygenfunction:: rascal_descriptor_indexes

.. .. doxygenfunction:: rascal_descriptor_densify

.. .. doxygenfunction:: rascal_descriptor_densify_values

.. ---------------------------------------------------------------------

.. .. doxygenenum:: rascal_indexes_kind

.. .. doxygenstruct:: rascal_indexes_t
..     :members:

.. .. doxygenstruct:: rascal_densified_position_t
..     :members:
