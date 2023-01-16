Installation
============

You can install equistore in the ways shown below depending
on which language you plan to use. If you want to
`build` equistore from the source code you will need a Rust compiler
besides a language specific compiler. You can install Rust using
`rustup <https://rustup.rs/>`_ or the package manager of your operating
system.

.. _install-python-lib:

Installing the Python library
-----------------------------

For building and using the Python package clone the repository
using `git <https://git-scm.com>`_ and install equistore using
`pip <https://pip.pypa.io>`_.

.. code-block:: bash

    git clone https://github.com/lab-cosmo/equistore
    cd equistore
    pip install .

Equistore is also provided as prebuilt wheel which avoids the intermediate
step of building the package with a Rust compiler from the source distribution.

.. code-block:: bash

    pip install --extra-index-url https://luthaf.fr/temporary-wheels/ equistore

.. _install-c-lib:

Installing the C and C++ library
--------------------------------

This installs a C-compatible shared library that can also be called from C++, as
well as CMake files that can be used with ``find_package(equistore)``.

.. code-block:: bash

    git clone https://github.com/lab-cosmo/equistore
    cd equistore
    mkdir build && cd build
    cmake <CMAKE_OPTIONS_HERE> ..
    cmake --build . --target install

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
    equistore = {git = "https://github.com/lab-cosmo/equistore.git"}
