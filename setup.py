import os
import sys
import uuid

from setuptools import setup
from setuptools.command.bdist_egg import bdist_egg


ROOT = os.path.realpath(os.path.dirname(__file__))
EQUISTORE_CORE = os.path.join(ROOT, "python", "equistore-core")
EQUISTORE_OPERATIONS = os.path.join(ROOT, "python", "equistore-operations")


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """

    def run(self):
        sys.exit(
            "Aborting implicit building of eggs. "
            + "Use `pip install .` or `python setup.py bdist_wheel && pip "
            + "uninstall equistore -y && pip install dist/equistore-*.whl` "
            + "to install from source."
        )


if __name__ == "__main__":
    sys.path.append(os.path.join(ROOT, "python", "equistore"))
    from version_from_git import git_extra_version

    version = "0.1.0" + git_extra_version()

    install_requires = []
    if os.path.exists(EQUISTORE_CORE):
        # we are building from a git checkout
        assert os.path.exists(EQUISTORE_OPERATIONS)

        # add a random uuid to the file url to prevent pip from using a cached
        # wheel for equistore-core, and force it to re-build from scratch
        uuid = uuid.uuid4()
        install_requires.append(
            f"equistore-core @ file://{EQUISTORE_CORE}?{uuid}",
        )
        install_requires.append(
            f"equistore-operations @ file://{EQUISTORE_OPERATIONS}?{uuid}",
        )
    else:
        # we are building from a sdist/installing from a wheel
        install_requires.append("equistore-core ~=0.1.0")
        install_requires.append("equistore-operations ~=0.1.0")

    setup(
        version=version,
        author=", ".join(open(os.path.join(ROOT, "AUTHORS")).read().splitlines()),
        install_requires=install_requires,
        cmdclass={
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
        },
    )
