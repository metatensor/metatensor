Installing equistore
====================

Equistore is available fro multiple programming languages, and how to install
and use it will depend on the programming language you are using.


If you want to build equistore from the source code you will need a Rust
compiler in addition to any language specific compiler. You can install Rust
using `rustup <https://rustup.rs/>`_ or the package manager of your operating
system.


.. _install-python-lib:

Installing the Python library
-----------------------------

For building and using the Python package clone the repository using `git
<https://git-scm.com>`_ and install equistore using `pip
<https://pip.pypa.io>`_.

From source:

.. code-block:: bash

    git clone https://github.com/lab-cosmo/equistore
    cd equistore
    pip install .

    # alternatively, the same thing in a single command
    pip install git+https://github.com/lab-cosmo/equistore


Equistore is also provided as prebuilt wheel which avoids the intermediate step
of building the package with a Rust compiler from the source code.

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
    cmake ..
    # configure cmake if needed
    cmake --build . --target install

The build and installation can be configures with a few cmake options, using
``-D<OPTION>=<VALUE>`` on the cmake command line, or one of the cmake GUI
(``cmake-gui`` or ``ccmake``). Here are the main configuration options:

+--------------------------------------+--------------------------------------------------------------------------------------+----------------+
| Option                               | Description                                                                          | Default        |
+======================================+======================================================================================+================+
| CMAKE_BUILD_TYPE                     | Type of build: debug or release                                                      | release        |
+--------------------------------------+--------------------------------------------------------------------------------------+----------------+
| CMAKE_INSTALL_PREFIX                 | Prefix in which the library will be installed                                        | ``/usr/local`` |
+--------------------------------------+--------------------------------------------------------------------------------------+----------------+
| INCLUDE_INSTALL_DIR                  | Path relative to ``CMAKE_INSTALL_PREFIX`` where the headers will be installed        | ``include``    |
+--------------------------------------+--------------------------------------------------------------------------------------+----------------+
| LIB_INSTALL_DIR                      | Path relative to ``CMAKE_INSTALL_PREFIX`` where the shared library will be installed | ``lib``        |
+--------------------------------------+--------------------------------------------------------------------------------------+----------------+
| BUILD_SHARED_LIBS                    | Default to installing and using a shared library instead of a static one             | ON             |
+--------------------------------------+--------------------------------------------------------------------------------------+----------------+
| EQUISTORE_INSTALL_BOTH_STATIC_SHARED | Install both the shared and static version of the library                            | ON             |
+--------------------------------------+--------------------------------------------------------------------------------------+----------------+


Using the Rust library
----------------------

Add the following to your project ``Cargo.toml``

.. code-block:: toml

    [dependencies]
    equistore = {git = "https://github.com/lab-cosmo/equistore.git"}
