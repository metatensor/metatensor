Labels
======

.. doxygenstruct:: eqs_labels_t
    :members:

The following functions operate on :c:type:`eqs_labels_t`:

- :c:func:`eqs_labels_create`: create the Rust-side data for the labels
- :c:func:`eqs_labels_clone`: increment the reference count of the Rust-side data
- :c:func:`eqs_labels_free`: decrement the reference count of the Rust-side data,
  and free the data when it reaches 0
- :c:func:`eqs_labels_position`: get the position of an entry in the labels
- :c:func:`eqs_labels_union`: get the union of two labels
- :c:func:`eqs_labels_intersection`: get the intersection of two labels

--------------------------------------------------------------------------------

.. doxygenfunction:: eqs_labels_create

.. doxygenfunction:: eqs_labels_clone

.. doxygenfunction:: eqs_labels_free

.. doxygenfunction:: eqs_labels_position

.. doxygenfunction:: eqs_labels_union

.. doxygenfunction:: eqs_labels_intersection
