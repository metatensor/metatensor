.. _engine-ipi:

i-PI
====

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatensor supported?
   * - https://ipi-code.org/
     - In the official version


Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

Only the :ref:`energy <energy-output>` output is supported, and used to run path
integral simulations, incorporating quantum nuclear effects in the statistical
sampling.

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

The metatensor interface is part of i-PI since version 3.0. Please refer to
`i-PI documentation`_ about how to install it.

.. _i-PI documentation: https://ipi-code.org/i-pi/getting-started.html#installing-i-pi

How to use the code
^^^^^^^^^^^^^^^^^^^

.. note::

  Here we assume you already have an exported model that you want to use in your
  simulations. Please see :ref:`this tutorial <atomistic-tutorial-export>` to
  learn how to manually create and export a model; or use a tool like
  `metatrain`_ to create a model based on existing architectures and your own
  dataset.

  .. _metatrain: https://github.com/metatensor/metatrain

The metatensor interface in i-PI provides a custom i-PI client that can be used
in combination with an i-PI server to run simulations. This client is managed
with the ``i-pi-driver-py`` command.

.. code-block:: bash

    # minimal version
    i-pi-driver-py -m metatensor -o template.xyz,model.pt

    # all possible options
    i-pi-driver-py -m metatensor -o template.xyz,model.pt,device=cpu,extensions=path/to/extensions,check_consistency=False

The minimal options to give to the ``metatensor`` client are the path to a
template structure for the simulated system (``template.xyz`` in the example
above) and the path to the metatensor model (``model.pt`` above). The template
structure must be a file that `ASE`_ can read. The code only uses it to get the
atomic types (assumed to be the atomic numbers) matching all particles in the
system.

The following options can also be specified using ``key=value`` syntax:

- **extensions**: the path to a folder containing TorchScript extensions. We
  will try to load any extension the model requires from there first;
- **device**: torch device to use to execute the model. Typical values would be
  ``cpu``, ``cuda``, ``cuda:2``, *etc.* By default, the code will find the best
  device for the model that is available on the current computer;
- **check_consistency**: whether to run some additional internal checks. Set
  this to ``True`` if you are seeing a strange behavior for a given model or
  when developing a new model.

.. seealso::

    You can use ``i-pi-driver-py --help`` to get all the options for the Python
    drivers.


.. _ASE: https://wiki.fysik.dtu.dk/ase/ase/io/io.html
