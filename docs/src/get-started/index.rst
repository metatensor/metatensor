Getting started
===============

.. TODO: expand this section

.. _install-python-lib:

Installing the Python library
-----------------------------

`equistore` provides Python bindings and that can be installed from pre-built wheels:

.. code-block:: bash

    pip install --extra-index-url https://luthaf.fr/temporary-wheels/ equistore

Installing from source is also easy, but you will need a complete build system, 
including  :ref:`install-rust`

.. code-block:: bash

    git clone https://github.com/lab-cosmo/equistore
    cd equistore
    pip install .

.. _install-c-lib:

Installing the C and C++ library
--------------------------------

From source:

.. code-block:: bash

    git clone https://github.com/lab-cosmo/equistore
    cd equistore
    mkdir build && cd build
    cmake ..
    # configure cmake if needed
    cmake --build . --target install


.. _install-rust:

Installing the Rust build system
--------------------------------

The core library is written in Rust, and so in order 
to compile and install the development version you need 
a functioning Rust installation. 

If you don't have one, you should be able to get up to 
speed quicky by following the instructions on the
`Rust installation page <https://www.rust-lang.org/tools/install>`_.

