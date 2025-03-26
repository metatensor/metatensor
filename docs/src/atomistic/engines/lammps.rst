.. _engine-lammps:

LAMMPS
======

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatensor supported?
   * - https://lammps.org
     - In a separate `fork <https://github.com/metatensor/lammps>`_


Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

Only the :ref:`energy <energy-output>` output is supported in LAMMPS, as a
custom ``pair_style``. This allows running molecular dynamics simulations with
interatomic potentials in metatensor format; distributing the simulation over
multiple nodes and potentially multiple GPUs.

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

The code is available in a custom fork of LAMMPS,
https://github.com/metatensor/lammps. We recommend you use `LAMMPS' CMake build
system`_ to configure the build.

To build metatensor-enabled LAMMPS, you'll need to provide the C++ version of
``libtorch``, either by installing PyTorch with a Python package manager
(``pip`` or ``conda``), or by downloading the right prebuilt version of the code
from https://pytorch.org/get-started/locally/. To build metatensor itself, you
will also need to install a Rust compiler and ``cargo`` installed, either with
`rustup`_ or the package manager of your operating system.

First, you should run the following code in a bash (or bash-compatible) shell to
get the code on your system and to teach CMake where to find ``libtorch``:

.. code-block:: bash

    # point this to the path where you extracted the C++ libtorch
    TORCH_PREFIX=<path/to/torch/installation>
    # if you used Python to install torch, you can do this:
    TORCH_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

    # patch a bug from torch's MKL detection
    git clone https://github.com/metatensor/lammps lammps-metatensor
    cd lammps-metatensor
    ./src/ML-METATENSOR/patch-torch.sh "$TORCH_PREFIX"

After what you can configure the build and compile the code:

.. code-block:: bash

    mkdir build && cd build

    # you can add more options here to enable other packages.
    cmake -DPKG_ML-METATENSOR=ON \
          -DLAMMPS_INSTALL_RPATH=ON \
          -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
          ../cmake

    cmake --build . --parallel 4 # or `make -jX`

    # optionally install the code on your machine. You can also directly use
    # the `lmp` binary in `lammps-metatensor/build/lmp` without installation
    cmake --build . --target install # or `make install`

By default, this code will try to find the metatensor libraries on your system
and use them. If cmake can not find the libraries, it will download and build
them as part of the main LAMMPS build. If you want, you can control this
behavior by adding `-DDOWNLOAD_METATENSOR=ON` to the cmake options to always
force a download or `-DDOWNLOAD_METATENSOR=OFF` to prevent any download.

.. _rustup: https://rustup.rs
.. _LAMMPS' CMake build system: https://docs.lammps.org/Build_cmake.html


How to use the code
^^^^^^^^^^^^^^^^^^^

.. note::

  Here we assume you already have an exported model that you want to use in your
  simulations. Please see :ref:`this tutorial <atomistic-tutorial-export>` to
  learn how to manually create and export a model; or use a tool like
  `metatrain`_ to create a model based on existing architectures and your own
  dataset.

  .. _metatrain: https://github.com/metatensor/metatrain

After building and optionally installing the code, you can now use ``pair_style
metatensor`` in your LAMMPS input files! Below is the reference documentation
for this pair style, following a similar structure to the official LAMMPS
documentation.

.. code-block:: shell

   pair_style metatensor model_path ... keyword values ...

* ``model_path`` = path to the file containing the exported metatensor model
* ``keyword`` = **device** or **extensions** or **check_consistency**

  .. parsed-literal::

       **device** values = device_name
         device_name = name of the Torch device to use for the calculations
       **extensions** values = directory
         directory = path to a directory containing TorchScript extensions as
         shared libraries. If the model uses extensions, we will try to load
         them from this directory first
       **check_consistency** values = on or off
         set this to on/off to enable/disable internal consistency checks,
         verifying both the data passed by LAMMPS to the model, and the data
         returned by the model to LAMMPS.

Examples
--------

.. code-block:: shell

   pair_style metatensor exported-model.pt device cuda extensions /home/user/torch-extensions/
   pair_style metatensor soap-gap.pt check_consistency on
   pair_coeff * * 6 8 1

Description
-----------

Pair style ``metatensor`` provides access to models following :ref:`metatensor's
atomistic models <atomistic-models>` interface; and enable using such models as
interatomic potentials to drive a LAMMPS simulation. The models can be fully
defined and trained by the user using Python code, or be existing pre-trained
models. The interface can be used with any type of machine learning model, as
long as the implementation of the model is compatible with TorchScript.

The only required argument for ``pair_style metatensor`` is the path to the model
file, which should be an exported metatensor model.

Optionally, users can define which torch ``device`` (e.g. cpu, cuda, cuda:0,
*etc.*) should be used to run the model. If this is not given, the code will run
on the best available device. If the model uses custom TorchScript operators
defined in a TorchScript extension, the shared library defining these extensions
will be searched in the ``extensions`` path, and loaded before trying to load
the model itself. Finally, ``check_consistency`` can be set to ``on`` or ``off``
to enable (respectively disable) additional internal consistency checks in the
data being passed from LAMMPS to the model and back.

A single ``pair_coeff`` command should be used with the ``metatensor`` style,
specifying the mapping from LAMMPS types to the atomic types the model can
handle. The first 2 arguments must be \* \* so as to span all LAMMPS atom types.
This is followed by a list of N arguments that specify the mapping of metatensor
atomic types to LAMMPS types, where N is the number of LAMMPS atom types.

Sample input file
-----------------

Below is a example input file that creates an FCC crystal of Nickel, and use a
metatensor model to run NPT simulations.

.. code-block:: bash

  units metal
  boundary p p p

  # create the simulation system without reading external data file
  atom_style atomic
  lattice fcc 3.6
  region box block 0 4 0 4 0 4
  create_box 1 box
  create_atoms 1 box

  labelmap atom 1 Ni
  mass Ni 58.693

  # define the interaction style to use the model in the "nickel-model.pt" file
  pair_style metatensor nickel-model.pt device cuda
  pair_coeff * * 28

  # simulation settings
  timestep 0.001 # 1fs timestep
  fix 1 all npt temp 243 243 $(100 * dt) iso 0 0 $(1000 * dt) drag 1.0

  # output setup
  thermo 10

  # run the simulation for 10000 steps
  run 10000
