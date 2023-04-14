import os
import sys
import uuid

from setuptools import setup
from setuptools.command.bdist_egg import bdist_egg


ROOT = os.path.realpath(os.path.dirname(__file__))
EQUISTORE_CORE = os.path.realpath(os.path.join(ROOT, "..", "equistore-core"))


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
    with open(os.path.join(ROOT, "AUTHORS")) as fd:
        authors = fd.read().splitlines()

    if authors[0].startswith(".."):
        raise RuntimeError(
            "AUTHORS file must be a symbolic link or the full file. "
            + "If you are building from Windows, enable git symlinks "
            + "(https://gist.github.com/huenisys/1efb64e57c37cfab7054c65702588fce) "
            + "and clone the code again."
        )

    sys.path.append(ROOT)
    from version_from_git import git_extra_version

    version = "0.1.0" + git_extra_version()

    install_requires = []
    if os.path.exists(EQUISTORE_CORE):
        # we are building from a git checkout

        # add a random uuid to the file url to prevent pip from using a cached
        # wheel for equistore-core, and force it to re-build from scratch
        uuid = uuid.uuid4()
        install_requires.append(f"equistore-core @ file://{EQUISTORE_CORE}?{uuid}")
    else:
        # we are building from a sdist/installing from a wheel
        install_requires.append("equistore-core ~=0.1.0")

    setup(
        version=version,
        author=", ".join(authors),
        install_requires=install_requires,
        cmdclass={
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
        },
    )
