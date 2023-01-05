By contributing to equistore, you accept and agree to the following terms and
conditions for your present and future contributions submitted to equistore.
Except for the license granted herein to equistore and recipients of software
distributed by equistore, you reserve all right, title, and interest in and to
your contributions.

Code of Conduct
---------------

As contributors and maintainers of equistore, we pledge to respect all people
who contribute through reporting issues, posting feature requests, updating
documentation, submitting merge requests or patches, and other activities.

We are committed to making participation in this project a harassment-free
experience for everyone, regardless of level of experience, gender, gender
identity and expression, sexual orientation, disability, personal appearance,
body size, race, ethnicity, age, or religion.

Examples of unacceptable behavior by participants include the use of sexual
language or imagery, derogatory comments or personal attacks, trolling, public
or private harassment, insults, or other unprofessional conduct.

Project maintainers have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are
not aligned to this Code of Conduct. Project maintainers who do not follow the
Code of Conduct may be removed from the project team.

This code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community.

.. Instances of abusive, harassing, or otherwise unacceptable behavior can be
.. reported by emailing xxx@xxx.org.

This Code of Conduct is adapted from the `Contributor Covenant`_, version 1.1.0,
available at https://contributor-covenant.org/version/1/1/0/

.. _`Contributor Covenant` : https://contributor-covenant.org

Getting involved
----------------

Contribution via merge requests are always welcome. Source code is available
from `Github`_. Before submitting a merge request, please open an issue to
discuss your changes. Use the only `master` branch for submitting your requests.

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
  1.56 to build equistore.
- **Python**: you can install ``Python`` and ``pip`` from your operating system.
  We require a Python version of at least 3.6.
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
    - ``cargo test --test=python-api`` (or ``tox`` directly) to run Python tests only;
    - ``cargo test --test=cpp-api`` to run the C/C++ API tests only;
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

The latter command ``tox -e format`` will use tox to do actual formatting
instead of just testing it.

.. _`cargo` : https://doc.rust-lang.org/cargo/

Contributing to the documentation
---------------------------------

The documentation of equistore is written in reStructuredText (rst) and uses the
`sphinx`_ documentation generator. In order to modify the documentation, first
create a local version of the code your machine as described above. Then, you
can build the documentation with:

.. code-block:: bash

    tox -e docs

You can visualise the local documentation with your favorite browser (here
Mozilla Firefox is used)

.. code-block:: bash

    firefox docs/build/html/index.html

.. _`sphinx` : https://www.sphinx-doc.org/en/master/
