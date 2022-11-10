Installation
============

The first step towards more accessible simulations starts here! To get started
with Equistore, you can install the latest stable version from one of the
sources below.

Precompiled Python installation
-------------------------------

TBD

.. _`install-python-lib`:

Python from source
------------------

You can build equistore from the source by::

    pip install https://github.com/lab-cosmo/equistore.git


.. _`install-c-lib`:

Installing the C/C++ library
----------------------------

This installs a C-compatible shared library that can also be called from C++, as
well as CMake files that can be used with ``find_package(rascaline)``.

.. code-block:: bash

    git clone https://github.com/lab-cosmo/equistore.git
    mkdir build
    cd build
    cmake <CMAKE_OPTIONS_HERE> ..
    make install

The build and installation can be configures with a few cmake options, using
``-D<OPTION>=<VALUE>`` on the cmake command line, or one of the cmake GUI
(``cmake-gui`` or ``ccmake``). Here are the main configuration options:

+--------------------------+--------------------------------------------------------------------------------------+----------------+
| Option                   | Description                                                                          | Default        |
+==========================+======================================================================================+================+
| CMAKE_BUILD_TYPE         | Type of build: debug or release                                                      | release        |
+--------------------------+--------------------------------------------------------------------------------------+----------------+
| CMAKE_INSTALL_PREFIX     | Prefix in which the library will be installed                                        | ``/usr/local`` |
+--------------------------+--------------------------------------------------------------------------------------+----------------+
| INCLUDE_INSTALL_DIR      | Path relative to ``CMAKE_INSTALL_PREFIX`` where the headers will be installed        | ``include``    |
+--------------------------+--------------------------------------------------------------------------------------+----------------+
| LIB_INSTALL_DIR          | Path relative to ``CMAKE_INSTALL_PREFIX`` where the shared library will be installed | ``lib``        |
+--------------------------+--------------------------------------------------------------------------------------+----------------+

Using the Rust library
----------------------

Add the following to your project ``Cargo.toml``

.. code-block:: toml

    [dependencies]
    rascaline = {git = "https://github.com/lab-cosmo/equistore.git"}
