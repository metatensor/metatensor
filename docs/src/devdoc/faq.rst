.. _devdoc-faq:

Developer FAQ
=============

This section addresses common questions and hurdles encountered when developing
with the ecosystem.

How do I use environment variables to control tests?
----------------------------------------------------

You can customize test execution by setting specific environment variables in your 
shell. This provides a lighter alternative to the :ref:`devdoc-local-ci` setup.
Please refer to :ref:`devdoc-get-started` for a full list of supported variables 
and their descriptions.

.. important::

   When running ``tox`` locally, your host Python version must be compatible
   with the ``torch`` version you request. For example, some versions of
   ``torch`` do not support Python 3.14.

How do I change the Python version used for local tests?
--------------------------------------------------------

If you encounter dependency resolution errors, ensure you are using a compatible
Python interpreter. You can use ``tox`` within a specific Python environment:

.. code-block:: bash

    # Use the -x flag to override the base_python for the environment
    METATENSOR_TESTS_TORCH_VERSION="2.2" \
    METATENSOR_TESTS_NUMPY_VERSION_PIN="<2.0" \
    tox -e core-tests -x testenv:core-tests.base_python=python3.11

How do I run the full CI suite locally?
---------------------------------------

You can use ``gh act`` to emulate GitHub Actions on your machine. This is
useful for catching environment-specific bugs before pushing code.
See :ref:`devdoc-local-ci` for setup and usage instructions.

.. note::

   These can be rather heavy for users without workstations. Using
   ``--action-offline-mode`` after the first run will help.

Often it makes sense for better logs to change ``tox.ini``, for instance to
remove `` --quiet`` from ``uv build``.

How do I run just the NumPy < 2.0 tests?
----------------------------------------

When using ``gh act``, you can isolate specific matrix configurations by
specifying the matrix keys. To run the legacy NumPy tests:

.. code-block:: bash

    gh act -j python-tests \
      --matrix os:ubuntu-24.04 \
      --matrix python-version:3.10 \
      --matrix torch-version:2.1 \
      --matrix numpy-version-pin:"<2.0"

.. tip::

   If you don't need the full Docker isolation of ``act``, you can also 
   replicate this legacy environment using :ref:`the environment variables method <devdoc-faq-env-vars>`.

.. _devdoc-faq-env-vars:

How do I run a single tox environment in CI?
--------------------------------------------

If you are using ``act`` and want to skip the full suite in favor of a
single environment (e.g., ``core-tests``), pass the ``TOXENV``
variable:

.. code-block:: bash

    gh act -j python-tests --env TOXENV=core-tests

How do I make a new release?
----------------------------

Follow a release PR. Find more `here <https://github.com/metatensor/metatensor/pulls?q=is%3Apr+release+is%3Aclosed>`_.

.. note::

   The changelog files are managed by hand.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Component
     - Release PR example
   * - **metatensor-core**
     - `v0.1.18 <https://github.com/metatensor/metatensor/pull/1013>`_
   * - **metatensor-rust**
     - `v0.2.3 <https://github.com/metatensor/metatensor/pull/1015>`_
   * - **metatensor-torch**
     - `v0.8.4 <https://github.com/metatensor/metatensor/pull/1042>`_
   * - **metatensor-operations**
     - `v0.3.0 <https://github.com/metatensor/metatensor/pull/779>`_
   * - **metatensor-learn**
     - `v0.4.0 <https://github.com/metatensor/metatensor/pull/1003>`_

How do I handle new PyTorch versions?
-------------------------------------

Follow a PyTorch upgrade PR. Then make a new release.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Complexity
     - Example
   * - Baseline version changes
     - `v2.9 <https://github.com/metatensor/metatensor/pull/1005>`_
   * - Handling deprecations
     - `v2.10 <https://github.com/metatensor/metatensor/pull/1041>`_

How do I organize my commits for a Pull Request?
------------------------------------------------

We prefer a clean history with logical commits. If your branch has many small
"fixup" or "wip" commits, you should squash them into a meaningful history.

.. note::

   This process is optional; if the changes are related to a single concept, we
   default to squashing during the merge.

- **Interactive Rebase**: To combine the last ``N`` commits, use:

  .. code-block:: bash

      git rebase -i HEAD~N

- **Soft Reset (The "Atomic" Method)**: If the history is too messy to rebase, reset to the main branch and re-add everything as logical blocks:

  .. code-block:: bash

      git reset --soft upstream/main
      git add -p  # selectively add changes
      git commit -m "logical commit message"
