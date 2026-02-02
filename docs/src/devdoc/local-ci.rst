.. _devdoc-local-ci:

Local CI Testing
================

Metatensor uses GitHub Actions (GHA) for continuous integration. While you can 
run tests directly via ``cargo`` or ``tox``, you may sometimes need to debug 
issues that only appear in the CI environment. You can emulate the GitHub 
Actions environment locally using `act <https://github.com/nektos/act>`_.

Installation
------------

We recommend using the GitHub CLI extension for ``act``:

1. Install the `GitHub CLI <https://cli.github.com/>`_.
2. Install the extension:

   .. code-block:: bash

       gh extension install nektos/gh-act

Running Matrix Jobs
-------------------

The metatensor test suite uses a matrix strategy to test multiple versions of 
Python, PyTorch, and NumPy. To run a specific configuration, use the 
``--matrix`` flag.

For example, to run the tests for Python 3.10 with NumPy < 2.0:

.. code-block:: bash

    gh act -j python-tests \
      --matrix os:ubuntu-24.04 \
      --matrix python-version:3.10 \
      --matrix torch-version:2.1 \
      --matrix numpy-version-pin:"<2.0"

Running Specific Tests
----------------------

To save time, you can instruct ``act`` to run only a specific ``tox`` 
environment by setting the ``TOXENV`` environment variable:

.. code-block:: bash

    gh act -j python-tests \
      --matrix os:ubuntu-24.04 \
      --matrix python-version:3.10 \
      --env TOXENV=core-tests
