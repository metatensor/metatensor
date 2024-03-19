import os

from sphinx.application import Sphinx


HERE = os.path.realpath(os.path.dirname(__file__))

if __name__ == "__main__":
    # the examples gallery is automatically generated upon the Sphinx object creation
    Sphinx(
        srcdir=os.path.join(HERE, "..", "src"),
        confdir=HERE,
        outdir=os.path.join(HERE, "..", "build"),
        doctreedir=os.path.join(HERE, "..", "build"),
        buildername="html",
    )
