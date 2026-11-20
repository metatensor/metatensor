.. _devdoc-faq:

Developer FAQ
=============

This section addresses common questions and hurdles encountered when developing
with the ecosystem.

How do I use environment variables to control tests?
----------------------------------------------------

You can customize test execution by setting specific environment variables in
your shell. Please refer to :ref:`running-tests` for a full list of supported
variables and their descriptions.

How do I change the Python version used for local tests?
--------------------------------------------------------

If you encounter dependency resolution errors, ensure you are using a compatible
Python interpreter. You can use ``tox`` within a specific Python environment:

.. code-block:: bash

    # Use the -x flag to override the base_python for the environment
    METATENSOR_TESTS_TORCH_VERSION="2.2" \
    tox -e core-tests -x testenv:core-tests.base_python=python3.11


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

For local development, consider rebuilding the enviroments:

.. code-block:: bash

    # remove compiled artifacts for new torch
    ./scripts/clean-python.sh
    # explicitly regenerate (or run tests directly)
    tox --recreate

This prevents errors along the lines of:

.. code-block:: bash

    found incompatible torch version: metatensor-torch was built against v2.9.1
        but we found v2.10.0

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

How do I run code coverage locally?
-----------------------------------

**Python coverage** is collected automatically by ``tox`` via ``pytest-cov``.
Each environment writes a ``.coverage`` file under ``.tox/<env>/``. To produce
a combined report:

.. code-block:: bash

    # run all (or specific) tox environments
    uvx tox
    # combine per-environment coverage files
    coverage combine .tox/*/.coverage
    # terminal report with missed lines
    coverage report --show-missing
    # or generate an HTML report
    coverage html
    firefox htmlcov/index.html

Path remapping is configured in the root ``pyproject.toml`` under
``[tool.coverage.paths]``.

**Rust coverage** uses ``cargo-llvm-cov`` with LLVM instrumentation. Install
the prerequisites first:

.. code-block:: bash

    rustup component add llvm-tools
    cargo +stable install cargo-llvm-cov --locked

Then generate an HTML report (including C/C++ code compiled via FFI):

.. code-block:: bash

    CC=$(which clang) CXX=$(which clang++) \
    LLVM_COV=$(which llvm-cov) LLVM_PROFDATA=$(which llvm-profdata) \
    LLVM_PROFILE_FILE="cargo-test-%p-%m.profraw" \
    CARGO_INCREMENTAL=0 RUSTFLAGS='-Cinstrument-coverage' \
    CFLAGS='-fprofile-instr-generate -fcoverage-mapping' \
    CXXFLAGS='-fprofile-instr-generate -fcoverage-mapping' \
    cargo llvm-cov --include-ffi --all-features --workspace --html

    # serve locally
    python -m http.server -d target/llvm-cov/html

The CI workflow in ``.github/workflows/coverage.yml`` uploads both
``coverage.xml`` (Python) and ``rust_codecov.json`` (Rust) to
`Codecov <https://app.codecov.io/gh/metatensor/metatensor>`_.
