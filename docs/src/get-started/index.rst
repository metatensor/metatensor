Getting started
===============

.. TODO: expand this section

.. _install-python-lib:

Installing the Python library
-----------------------------

From source:

.. code-block:: bash

    git clone https://github.com/lab-cosmo/equistore
    cd equistore
    pip install .

Pre-built wheels:

.. code-block:: bash

    pip install --extra-index-url https://luthaf.fr/temporary-wheels/ equistore

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
