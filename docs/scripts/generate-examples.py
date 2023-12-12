import os

from sphinx.application import Sphinx


HERE = os.path.realpath(os.path.dirname(__file__))

if __name__ == "__main__":
    # the examples gallery is automatically generated upon the Sphinx object creation
    _ = Sphinx(
        srcdir=os.path.join(HERE, "..", "src"),
        confdir=os.path.join(HERE, "sphinx-gallery"),
        outdir=os.path.join(HERE, "..", "build"),
        doctreedir=os.path.join(HERE, "..", "build"),
        buildername="html",
    )
