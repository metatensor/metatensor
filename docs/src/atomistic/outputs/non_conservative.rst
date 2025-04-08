.. _non-conservative-forces-output:

Non-conservative forces
^^^^^^^^^^^^^^^^^^^^^^^

Non-conservative forces are forces that are not calculated as the negative
gradient of the potential energy. These are generally faster than forces
computed from the potential energy by backpropagation. However, these
predictions must be used with care, see https://arxiv.org/abs/2412.11569.

In metatensor atomistic models, they are associated with the
``"non_conservative_forces"`` key in the model outputs,
and must adhere to the following metadata:

.. list-table:: Metadata for non-conservative forces output
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single
      entry set to ``0``. Non-conservative forces are always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]``
    - the samples should be named ``["system", "atom"]``, since
      non-conservative forces are always per-atom.

      ``"system"`` must range from 0 to the number of systems given as input to
      the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    - ``"xyz"``
    - non-conservative forces must have a single component dimension
      named ``"xyz"``, with three entries set to ``0``, ``1``, and ``2``.
      The non-conservative forces are always 3D vectors, and the order of the components
      is x, y, z.

  * - properties
    - ``"non_conservative_forces"``
    - non-conservative forces must have a single property dimension named
      ``"non_conservative_forces"``, with a single entry set to ``0``.

The following simulation engines can use the ``"non_conservative_forces"`` output:

.. grid:: 1 1 1 1

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    .. py:currentmodule:: metatensor.torch.atomistic.ase_calculator.MetatensorCalculator

    |ase-logo|

    (using :py:meth:`run_model`)


.. _non-conservative-stress-output:

Non-conservative stress
^^^^^^^^^^^^^^^^^^^^^^^

The "non-conservative stress" is a stress tensor that is not calculated using
derivatives of the potential energy. This is generally faster than computing the
stress from the potential energy by backpropagation. However, these predictions
must be used with care, see https://arxiv.org/abs/2412.11569.

In metatensor atomistic models, it is associated with the ``"non_conservative_stress"``
key in the model outputs, and must adhere to the following metadata:

.. list-table:: Metadata for non-conservative stress output
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single
      entry set to ``0``. Non-conservative forces are always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``"system"``
    - the samples should contain a single sample named ``"system"``

      The values must range from 0 to the number of systems given as input to the model.

  * - components
    - ``["xyz_1"], ["xyz_2"]``
    - the non-conservative stress must have two components labels with ``"xyz_1"`` and
      ``"xyz_2"`` as their respective names, both with three entries set to ``0``,
      ``1``, and ``2``. The order of the components along both directions is x, y, z.

  * - properties
    - ``"non_conservative_stress"``
    - the non-conservative stress must have a single property dimension named
      ``"non_conservative_stress"``, with a single entry set to ``0``.

The following simulation engines can use the ``"non_conservative_stress"`` output:

.. grid:: 1 1 1 1

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    .. py:currentmodule:: metatensor.torch.atomistic.ase_calculator.MetatensorCalculator

    |ase-logo|

    (using :py:meth:`run_model`)
