import os
import subprocess
import sys
from datetime import datetime

import toml
from sphinx_gallery.sorting import FileNameSortKey


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# -- Project information -----------------------------------------------------

project = "metatensor"
author = ", ".join(open(os.path.join(ROOT, "AUTHORS")).read().splitlines())
copyright = f"{datetime.now().date().year}, {author}"


def load_version_from_cargo_toml():
    with open(os.path.join(ROOT, "metatensor-core", "Cargo.toml")) as fd:
        data = toml.load(fd)
    return data["package"]["version"]


# The full version, including alpha/beta/rc tags
release = load_version_from_cargo_toml()


def build_doxygen_docs():
    try:
        os.mkdir(os.path.join(ROOT, "docs", "build"))
    except OSError:
        pass

    # we need to run a cargo build to make sure the header is up to date
    subprocess.run(["cargo", "build"])
    subprocess.run(["doxygen", "Doxyfile"], cwd=os.path.join(ROOT, "docs"))


def generate_examples():
    # we can not run sphinx-gallery in the same process as the normal sphinx, since they
    # need to import metatensor.torch differently (with and without
    # METATENSOR_IMPORT_FOR_SPHINX=1). So instead we run it inside a small script, and
    # include the corresponding output later.
    script = os.path.join(ROOT, "docs", "scripts", "generate-examples.py")
    subprocess.run([sys.executable, script], capture_output=False)


def setup(app):
    build_doxygen_docs()
    generate_examples()

    app.add_css_file("css/metatensor.css")

    # when importing metatensor-torch, this will change the definition of the classes
    # to include the documentation
    os.environ["METATENSOR_IMPORT_FOR_SPHINX"] = "1"


# -- General configuration ---------------------------------------------------

needs_sphinx = "4.0.0"

python_use_unqualified_type_names = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_toggleprompt",
    "sphinx_gallery.gen_gallery",
    "breathe",
    "myst_parser",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    "examples/index.rst",
    "sg_execution_times.rst",
]

sphinx_gallery_conf = {
    "filename_pattern": ".*",
    "examples_dirs": [
        os.path.join(ROOT, "python", "examples", "learn"),
        os.path.join(ROOT, "python", "examples", "atomistic"),
    ],
    "gallery_dirs": [
        os.path.join("examples", "learn"),
        os.path.join("examples", "atomistic"),
    ],
    # Make the code snippet for metatensor functions clickable
    "reference_url": {"metatensor": None},
    "prefer_full_module": [
        "metatensor",
        r"metatensor\.learn\.data",
    ],
    "within_subsection_order": FileNameSortKey,
}


autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "short"

breathe_projects = {
    "metatensor": os.path.join(ROOT, "docs", "build", "doxygen", "xml"),
}
breathe_default_project = "metatensor"
breathe_domain_by_extension = {
    "h": "c",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "rascaline": ("https://luthaf.fr/rascaline/latest/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../static"]

html_js_files = [os.path.join("js", "custom.js")]
