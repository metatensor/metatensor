Contribution via merge requests are always welcome. Source code is available
from `Github`_. Before submitting a merge request, please open an issue to
discuss your changes. Use the only `master` as reference branch for submitting
your requests.

.. _`Github` : https://github.com/lab-cosmo/equistore

Required tools
--------------

You will need to install and get familiar with the following tools when working
on equistore:

- **git**: the software we use for version control of the source code. See
  https://git-scm.com/downloads for installation instructions.
- **the rust compiler**: you will need both ``rustc`` (the compiler) and
  ``cargo`` (associated build tool). You can install both using `rustup`_, or
  use a version provided by your operating system. We need at least Rust version
  1.61 to build equistore.
- **Python**: you can install ``Python`` and ``pip`` from your operating system.
  We require a Python version of at least 3.7.
- **tox**: a Python test runner, cf https://tox.readthedocs.io/en/latest/. You
  can install tox with ``pip install tox``.

Additionally, you will need to install the following software, but you should
not have to interact with them directly:

- **cmake**: we need a cmake version of at least 3.10.
- **a C++ compiler** we need a compiler supporting C++11. GCC >= 5, clang >= 3.7
  and MSVC >= 15 should all work, although MSVC has not been tested yet.

.. _rustup: https://rustup.rs
.. _tox: https://tox.readthedocs.io/en/latest

Getting the code
----------------

The first step when developing equistore is to `create a fork`_ of the main
repository on github, and then clone it locally:

.. code-block:: bash

    git clone <insert/your/fork/url/here>
    cd equistore

    # setup the local repository so that the master branch tracks changes in
    # the main repository
    git remote add upstream https://github.com/lab-cosmo/equistore
    git fetch upstream
    git branch master --set-upstream-to=upstream/master

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

    cd <path/to/equistore/repo>
    cargo test  # or cargo test --release to run tests in release mode

These are exactly the same tests that will be performed online in our Github CI
workflows. You can also run only a subset of tests with one of these commands:

- ``cargo test`` runs everything
- ``cargo test --test=<test-name>`` to run only the tests in ``tests/<test-name>.rs``;
    - ``cargo test --test=python-api`` (or ``tox`` directly, see below) to run
      Python tests only;
    - ``cargo test --test=c-api-tests`` to run the C/C++ API tests only. For these
      tests, if `valgrind`_ is installed, it will be used to check for memory
      errors. You can disable this by setting the `EQUISTORE_DISABLE_VALGRIND`
      environment variable to 1 (`export EQUISTORE_DISABLE_VALGRIND=1` for most
      Linux/macOS shells);
- ``cargo test --lib`` to run unit tests;
- ``cargo test --doc`` to run documentation tests;
- ``cargo bench --test`` compiles and run the benchmarks once, to quickly ensure
  they still work.

You can add some flags to any of above commands to further refine which tests
should run:

- ``--release`` to run tests in release mode (default is to run tests in debug mode)
- ``-- <filter>`` to only run tests whose name contains filter, for example ``cargo test -- keys_to_properties``

Also, you can run individual python tests using `tox`_ if you wish to test only
specific functionalities, for example:

.. code-block:: bash

    tox -e tests         # unit tests
    tox -e lint          # code style
    tox -e build-python  # python packaging

    tox -e format        # format all files

The last command ``tox -e format`` will use tox to do actual formatting instead
of just checking it.

You can run only a subset of the tests with ``tox -e tests -- <test/file.py>``,
replacing ``<test/file.py>`` with the path to the files you want to test, e.g.
``tox -e tests -- python/tests/operations/abs.py``.

.. _`cargo` : https://doc.rust-lang.org/cargo/
.. _valgrind: https://valgrind.org/

Contributing to the documentation
---------------------------------

The documentation of equistore is written in reStructuredText (rst) and uses the
`sphinx`_ documentation generator. In order to modify the documentation, first
create a local version of the code on your machine as described above. Then, you
can build the documentation with:

.. code-block:: bash

    tox -e docs

You can visualize the local documentation with your favorite browser (here
Mozilla Firefox is used)

.. code-block:: bash

    firefox docs/build/html/index.html

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
        values ``67``. You can link to classes :py:class:`equistore.Labels`. This
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
        >>> from equistore import func
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
