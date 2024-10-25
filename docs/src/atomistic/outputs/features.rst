.. _features-output:

Features
^^^^^^^^

Features are numerical vectors representing a given structure or
atom/atom-centered environment in an abstract n-dimensional space. They are also
sometimes called descriptors, representations, embedding, *etc.*

Features can be computed with some analytical expression (for example `SOAP
power spectrum`_, `atom-centered symmetry functions`_, …), or learned internally
by a neural-network or a similar architecture.

.. _SOAP power spectrum: https://doi.org/10.1103/PhysRevB.87.184115
.. _Atom-centered symmetry functions: https://doi.org/10.1063/1.3553717

In metatensor atomistic models, they are associated with the ``"features"`` key
in the model outputs, and must adhere to the following metadata:

.. list-table:: Metadata for features output
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the features keys must have a single dimension named ``"_"``, with a single
      entry set to ``0``. The feature is always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]`` or ``["system"]``
    - the samples should be named ``["system", "atom"]`` for per-atom outputs;
      or ``["system"]`` for global outputs.

      The ``"system"`` index should always be 0, and the ``"atom"`` index should
      be the index of the atom (between 0 and the total number of atoms). If
      ``selected_atoms`` is provided, then only the selected atoms for each
      system should be part of the samples.

  * - components
    -
    - the features must not have any components.

  * - properties
    -
    - the features can have arbitrary properties.

.. note::
  Features are typically handled without a unit, so the ``"unit"`` field of
  :py:func:`metatensor.torch.atomistic.ModelOutput` is mainly left empty.

The following simulation engines can use the ``"features"`` output:

.. grid:: 1 3 3 3

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-plumed
    :link-type: ref

    |plumed-logo|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    .. py:currentmodule:: metatensor.torch.atomistic.ase_calculator.MetatensorCalculator

    |ase-logo|

    (using :py:meth:`run_model`)

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-chemiscope
    :link-type: ref

    |chemiscope-logo|

Features gradients
------------------

As for the :ref:`energy <energy-output-gradients>`, features are typically used
with automatic gradient differentiation. Explicit gradients could be allowed if
you have a use case for them, but are currently not until they are fully
specified.
