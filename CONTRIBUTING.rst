Contributions via pull requests are always welcome. Source code is available from
`GitHub`_. Before submitting a pull request, please open an issue to discuss
your changes. Use only the `main` branch as the target branch when submitting a pull request (PR).

.. _`GitHub` : https://github.com/metatensor/metatensor

Interactions with the metatensor projects must follow our `code of conduct`_.

.. _code of conduct: https://github.com/metatensor/metatensor/blob/main/CODE_OF_CONDUCT.md

Required tools
--------------

You will need to install and get familiar with the following tools when working
on metatensor:

- **git**: the software we use for version control of the source code. See
  https://git-scm.com/downloads for installation instructions.
- **the rust compiler**: you will need both ``rustc`` (the compiler) and
  ``cargo`` (associated build tool). You can install both using `rustup`_, or
  use a version provided by your operating system. We need at least Rust version
  1.74 to build metatensor.
- **Python**: you can install ``Python`` and ``pip`` from your operating system.
  We require a Python version of at least 3.9.
- **tox**: a Python test runner, see https://tox.readthedocs.io/en/latest/. You
  can install tox with ``pip install tox``.

Additionally, you will need to install the following software, but you should
not have to interact with them directly:

- **CMake**: we need a CMake version of at least 3.16.
- **a C++ compiler** we need a compiler supporting C++11 (C++17 for
  ``metatensor-torch``). GCC >= 7, clang >= 5 and MSVC >= 19 should all work.

.. _rustup: https://rustup.rs
.. _tox: https://tox.readthedocs.io/en/latest

.. admonition:: Optional tools

  Depending on which part of the code you are working on, you might experience a
  lot of time spend re-compiling Rust or C++ code, even if you did not change
  them. If you'd like faster builds (and in turn faster tests), you can use
  `sccache`_ or the classic `ccache`_ to only re-run the compiler if the
  corresponding source code changed. To do this, you should install and configure
  one of these tools (we suggest sccache since it also supports Rust), and then
  configure cmake and cargo to use them by setting environnement variables. On
  Linux and macOS, you should set the following (look up how to do set environment
  variable with your shell):

  .. code-block:: bash

      CMAKE_C_COMPILER_LAUNCHER=sccache
      CMAKE_CXX_COMPILER_LAUNCHER=sccache
      # only if you have sccache and not ccache
      RUSTC_WRAPPER=sccache


  .. _sccache: https://github.com/mozilla/sccache
  .. _ccache: https://ccache.dev/

Getting the code
----------------

The first step when developing metatensor is to `create a fork`_ of the main
repository on github, and then clone it locally:

.. code-block:: bash

    git clone <insert/your/fork/url/here>
    cd metatensor

    # setup the local repository so that the main branch tracks changes in
    # the original repository
    git remote add upstream https://github.com/metatensor/metatensor
    git fetch upstream
    git branch main --set-upstream-to=upstream/main

Once you get the code locally, you will want to run the tests to check
everything is working as intended. See the next section on this subject.

If everything is working, you can create your own branches to work on your
changes:

.. code-block:: bash

    git checkout -b <my-branch-name>
    # code code code

    # push your branch to your fork
    git push -u origin <my-branch-name>
    # follow the link in the message to open a pull request (PR)

.. _create a fork: https://docs.github.com/en/github/getting-started-with-github/fork-a-repo

Running tests
-------------

The continuous integration pipeline is based on `cargo`_. You can run all tests
with:

.. code-block:: bash

    cd <path/to/metatensor/repo>
    cargo test  # or cargo test --release to run tests in release mode

These are exactly the same tests that will be performed online in our GitHub CI
workflows. You can also run only a subset of tests with one of these commands:

- ``cargo test`` runs everything
- ``cargo test --package=metatensor-core`` to run the C/C++ tests only;

  - ``cargo test --test=run-cxx-tests`` will run the unit tests for the C/C++
    API.
  - ``cargo test --test=check-cxx-install`` will build the C/C++ interfaces,
    install them and the associated CMake files and then try to build a basic
    project depending on this interface with CMake;

- ``cargo test --package=metatensor-torch`` to run the C++ TorchScript extension
  tests only;

  - ``cargo test --test=run-torch-tests`` will run the unit tests for the
    TorchScript C++ extension;
  - ``cargo test --test=check-cxx-install`` will build the C++ TorchScript
    extension, install it and then try to build a basic project depending on
    this extension with CMake;

- ``cargo test --package=metatensor-python`` (or ``tox`` directly, see below) to
  run Python tests only;
- ``cargo test --lib`` to run unit tests;
- ``cargo test --doc`` to run documentation tests;
- ``cargo bench --test`` compiles and run the benchmarks once, to quickly ensure
  they still work.

You can add some flags to any of above commands to further refine which tests
should run:

- ``--release`` to run tests in release mode (default is to run tests in debug mode)
- ``-- <filter>`` to only run tests whose name contains filter, for example ``cargo test -- keys_to_properties``

Also, you can run individual Python tests using `tox`_ if you wish to run a
subset of Python tests, for example:

.. code-block:: bash

    tox -e core-tests                     # unit tests for metatensor-core
    tox -e operations-numpy-tests       # unit tests for metatensor-operations without torch
    tox -e operations-torch-tests         # unit tests for metatensor-operations with torch
    tox -e torch-tests                    # unit tests for metatensor-torch
    tox -e docs-tests                     # doctests (checking inline examples) for all packages
    tox -e lint                           # code style
    tox -e build-python                   # python packaging

    tox -e format                         # format all files

The last command ``tox -e format`` will use tox to do actual formatting instead
of just checking it, you can use to automatically fix some of the issues
detected by ``tox -e lint``.

You can run only a subset of the tests with ``tox -e tests -- <test/file.py>``,
replacing ``<test/file.py>`` with the path to the files you want to test, e.g.
``tox -e tests -- python/tests/operations/abs.py``.

To get the release build for ``tox`` runs, set the environment variable.

.. code-block:: bash

    METATENSOR_BUILD_TYPE="release" tox -e core-tests

This corresponds to running ``cargo test --package-metatensor-python --release``
but on the subset of interest.

Controlling tests behavior with environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are a handful of environment variables that you can set to control the
behavior of tests:

- ``METATENSOR_BUILD_TYPE=release`` will use the ``release`` target for the
  rust libraries, and is useful for ``tox``;
- ``METATENSOR_DISABLE_VALGRIND=1`` will disable the use of `valgrind`_ for the
  C++ tests. Valgrind is a tool that check for memory errors in native code, but
  it makes the tests run quite a bit slower;
- ``METATENSOR_TESTS_TORCH_VERSION`` allow you to run the tests against a
  specific PyTorch version instead of the latest one. For example, setting it to
  ``METATENSOR_TESTS_TORCH_VERSION=1.13`` will run the tests against PyTorch
  1.13;
- ``PIP_EXTRA_INDEX_URL`` can be used to pull PyTorch (or other dependencies)
  from a different index. This can be useful on Linux if you have issues with
  CUDA, since the default PyTorch version expects CUDA to be available. A
  possible workaround is to use the CPU-only version of PyTorch in the tests, by
  setting ``PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu``;
- ``PYTORCH_JIT=0`` can be used to disable Python to TorchScript compilation of
  code; producing error messages which should be easier to understand.

.. _`cargo` : https://doc.rust-lang.org/cargo/
.. _valgrind: https://valgrind.org/


Code coverage
~~~~~~~~~~~~~

The code coverage is reported at `codecov`_. Locally, coverage reports can be
generated for each programming language.

.. _codecov: https://codecov.io/gh/metatensor/metatensor

Python coverage
^^^^^^^^^^^^^^^

Python coverage is written out as several individual files. It is easier to
combine all reports and open the generated ``html`` file in a browser

.. code-block:: bash

    tox
    coverage combine .tox/*/.coverage
    coverage html
    firefox htmlcov/index.html


Rust coverage
^^^^^^^^^^^^^

Rust coverage is instrumeted through the `cargo-llvm-cov`_ plugin

.. code-block:: bash

    rustup component add llvm-tools
    cargo +stable install cargo-llvm-cov --locked

It is desirable to generate coverage taking into account the coverage for the bindings to C/C++ and Python so:

.. code-block:: bash

    CC=$(which clang) CXX=$(which clang++) \
    LLVM_COV=$(which llvm-cov) LLVM_PROFDATA=$(which llvm-profdata) \
    LLVM_PROFILE_FILE="cargo-test-%p-%m.profraw" \
    CARGO_INCREMENTAL=0 RUSTFLAGS='-Cinstrument-coverage' \
    CFLAGS='-fprofile-instr-generate -fcoverage-mapping' \
    CXXFLAGS='-fprofile-instr-generate -fcoverage-mapping' \
    cargo llvm-cov --include-ffi --all-features \
    --workspace --html

Finally a local ``http`` server can be used to view the generated ``html``


.. code-block:: bash

    python -m http.server -d target/llvm-cov/html

.. _cargo-llvm-cov: https://github.com/taiki-e/cargo-llvm-cov

Contributing to the documentation
---------------------------------

The documentation of metatensor is written in reStructuredText (rst) and uses the
`sphinx`_ documentation generator. In order to modify the documentation, first
create a local version of the code on your machine as described above. Then, you
can build the documentation with:

.. code-block:: bash

    tox -e docs

In addition to have the requirements listed above, you will also need to install doxygen (e.g. ``apt install doxygen`` on Debian-based systems).

You can then visualize the local documentation with your favorite browser with
the following command (or open the :file:`docs/build/html/index.html` file
manually).

.. code-block:: bash

    # on linux, depending on what package you have installed:
    xdg-open docs/build/html/index.html
    firefox docs/build/html/index.html

    # on macOS:
    open docs/build/html/index.html

.. _`sphinx` : https://www.sphinx-doc.org/en/master/

Python doc strings
~~~~~~~~~~~~~~~~~~

Our docstring format follows the `sphinx format`_ and a typical function doc string
looks like the following.

.. code-block:: python

    def func(value_1: float, value_2: int) -> float:
        r"""A one line summary sentence of the function.

        Extensive multi-line summary of what is going in. Use single
        backticks for parameters of the function like `width` and two ticks for
        values ``67``. You can link to classes :py:class:`metatensor.Labels`. This
        also works for other classes and functions like :py:obj:`True`.

        Inline Math is also possible with :math:`\mathsf{R}`. Or as a math block.

        .. math::

            \mathbf{x}' = \mathsf{R}\mathbf{x}


        :param value_1:
            The first parameter of the function, a :py:class:`float`.
        :param value_2:
            The second parameter of the function, an :py:class:`int`.

        :returns result:
            The result of the calculation, a :py:class:`float`.

        :raises TypeError:
            If `value_1` is not a :py:class:`float` or `value_2` is not a :py:class:`int`.
        :raises ValueError:
            If `value_1` is not greater than zero.

        Examples
        --------
        >>> from metatensor import func
        >>> func(1, 1)
        42
        """
        ...
        return result

Guidelines for writing Python doc strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use Python typing in the function arguments, indicate return types.

* Start the description after each ``:param:`` or ``:return:`` in a new line and add an
  empty line between the parameter and return block.

* Emphasize function and class parameters with a single backtick i.e ```param``` and
  general variables should be double backticked . i.e. ````my_variable````

* If you include any maths, make the string a
  `raw string`_ by prefixing with ``r``, e.g.,

  .. code-block:: python

    r"""Some math like :math:`\nu^2 / \rho` with backslashes."""

  Otherwise the ``\n`` and ``\r`` will be rendered as ASCII escape sequences that break
  lines without you noticing it or you will get either one of the following two
  errors message

  1. `Explicit markup ends without a blank line; unexpected unindent`
  2. `Inline interpreted text or phrase reference start-string without end string`

* The examples are tested with `doctest`_. Therefore, please make sure that they are
  complete and functioning (with all required imports).
  Use the ``>>>`` syntax for inputs (followed by ``...`` for multiline inputs) and no
  indentation for outputs for the examples.

  .. code-block:: python

      """
      >>> a = np.array(
      ...    [1, 2, 3, 4]
      ... )
      """

.. _`sphinx format` : https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
.. _`raw string` : https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals
.. _`doctest` : https://docs.python.org/3/library/doctest.html

Guidelines for writing Rust additions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although the library itself forms best-practice guides for new additions, some
other unstructured notes might be of use:

- Do not use ``rustfmt`` for any file other than the auto-generated
  ``c_api.rs``.

Useful developer scripts
------------------------

The following scripts can be useful to developers:

- ``./scripts/clean-python.sh``: remove all generated files related to Python,
  including all build caches
- ``./scripts/update-declarations.sh``: update API declaration in Python, Rust
  and Julia from the latest version of the ``metatensor.h`` header. This should
  be used after any change to the C API.
