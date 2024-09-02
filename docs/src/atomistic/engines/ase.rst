.. _engine-ase:

ASE
===

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatensor supported?
   * - https://wiki.fysik.dtu.dk/ase/
     - As part of the ``metatensor-torch`` package

Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

.. py:currentmodule:: metatensor.torch.atomistic.ase_calculator

- the :ref:`energy <energy-output>` output is supported and fully integrated
  with ASE calculator interface (i.e. :py:meth:`ase.Atoms.get_potential_energy`,
  :py:meth:`ase.Atoms.get_forces`, …);
- arbitrary outputs can be computed for any :py:class:`ase.Atoms` using
  :py:meth:`MetatensorCalculator.run_model`;

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

The code is available in the ``metatensor-torch`` package, in the
:py:class:`metatensor.torch.atomistic.ase_calculator.MetatensorCalculator`
class.

How to use the code
^^^^^^^^^^^^^^^^^^^

See the :ref:`corresponding tutorial <atomistic-tutorial-md>`, and API
documentation of the :py:class:`MetatensorCalculator` class.
