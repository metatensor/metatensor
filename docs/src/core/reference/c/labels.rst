Labels
======

.. doxygentypedef:: mts_labels_t

The following functions operate on :c:type:`mts_labels_t`:

- :c:func:`mts_labels`: create new labels from dimension names and a values array
- :c:func:`mts_labels_assume_unique`: create new labels from a values array without verifying uniqueness
- :c:func:`mts_labels_clone`: increment the reference count of the labels
- :c:func:`mts_labels_free`: decrement the reference count of the labels
- :c:func:`mts_labels_dimensions`: get the dimension names of the labels
- :c:func:`mts_labels_values`: get the values of the labels as an array
- :c:func:`mts_labels_values_cpu`: get the values of the labels on CPU
- :c:func:`mts_labels_position`: get the position of an entry in the labels
- :c:func:`mts_labels_union`: get the union of two labels
- :c:func:`mts_labels_intersection`: get the intersection of two labels
- :c:func:`mts_labels_difference`: get the set difference of two labels
- :c:func:`mts_labels_select`: select entries in labels that match a selection

--------------------------------------------------------------------------------

.. doxygenfunction:: mts_labels

.. doxygenfunction:: mts_labels_assume_unique

.. doxygenfunction:: mts_labels_clone

.. doxygenfunction:: mts_labels_free

.. doxygenfunction:: mts_labels_dimensions

.. doxygenfunction:: mts_labels_values

.. doxygenfunction:: mts_labels_values_cpu

.. doxygenfunction:: mts_labels_position

.. doxygenfunction:: mts_labels_union

.. doxygenfunction:: mts_labels_intersection

.. doxygenfunction:: mts_labels_difference

.. doxygenfunction:: mts_labels_select
