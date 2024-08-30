.. _atomistic-models-outputs:

Standard model outputs
======================

In order for multiple simulation engines to be able use arbitrary metatensor
atomistic models to compute atomic properties, we need all the models to return
the same metadata for a given output. If your model returns one of the output
defined in this documentation, then it should follow the metadata structure
described here.

For other kind of output, you are free to use any relevant metadata structure,
but if multiple people are producing the same kind of outputs, they are
encouraged to come together, define the metadata they need and add a new section
to these pages.

.. toctree::
  :maxdepth: 1
  :hidden:

  energy
  features


Physical quantities
^^^^^^^^^^^^^^^^^^^

The first set of standardized outputs in metatensor atomistic models are
physical quantities, i.e. quantities with a well-defined physical meaning.

.. grid:: 1 2 2 2

    .. grid-item-card:: Energy
      :link: energy-output
      :link-type: ref

      .. image:: /../static/images/energy-output.png

      The potential energy associated with a given system conformation. This
      can be used to run molecular simulations based on machine learning
      interatomic potentials.

    .. grid-item-card:: Energy ensemble
      :link: energy-ensemble-output
      :link-type: ref

      .. image:: /../static/images/energy-ensemble-output.png

      An ensemble of multiple potential energies predictions, when running
      multiple models simultaneously.


Machine learning outputs
^^^^^^^^^^^^^^^^^^^^^^^^

The first set of standardized outputs in metatensor atomistic models are
specific to machine learning and related tools.

.. grid:: 1 2 2 2

    .. grid-item-card:: Features
      :link: features-output
      :link-type: ref

      .. image:: /../static/images/features-output.png

      Features are numerical vectors representing a given structure or atomic
      environment in an abstract n-dimensional space.
