.. _torch-cxx-api-reference:

TorchScript C++ API reference
=============================

To use equistore's TorchScript C++ API from your own code, you should ``#include
<equistore/torch.hpp>`` and use the functions and classes documented below. You
should then link with libtorch, and the both the ``equistore_torch`` and
``equistore`` shared libraries.

The easiest way to link to everything and set the right compiler flags is to use
CMake after :ref:`installing equistore's C++ Torch interface
<install-torch-script>`:

.. code-block:: cmake

    # This will find equistore_torch, equistore and Torch on your system
    find_package(equistore_torch)

    # add executables/libraries
    add_executable(MyExecutable my_sources.cxx)
    add_library(MyLibrary my_sources.cxx)

    # Link to equistore_torch, this makes the headers accessible
    # and link to the right libraries
    target_link_libraries(MyExecutable equistore_torch)
    target_link_libraries(MyLibrary equistore_torch)

.. toctree::
    :maxdepth: 1

    tensor
    block
    labels
    serialization
