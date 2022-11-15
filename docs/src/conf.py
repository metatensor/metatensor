import os
import shutil
import subprocess
import sys
from datetime import datetime

import toml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.append(os.path.join(ROOT, "python", "src"))

# -- Project information -----------------------------------------------------

project = "Equistore"
copyright = f"{datetime.now().date().year}, Equistore developers"
author = "Equistore developers"


def load_version_from_cargo_toml():
    with open(os.path.join(ROOT, "Cargo.toml")) as fd:
        data = toml.load(fd)
    return data["package"]["version"]


# The full version, including alpha/beta/rc tags
release = load_version_from_cargo_toml()


def build_cargo_docs():
    subprocess.run(["cargo", "doc", "--no-deps"])
    output_dir = os.path.join(ROOT, "docs", "build", "html", "reference", "rust")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(
        os.path.join(ROOT, "target", "doc"),
        output_dir,
    )


def build_doxygen_docs():
    # we need to run a build to make sure the header is up to date
    subprocess.run(["cargo", "build"])
    subprocess.run(["doxygen", "Doxyfile"], cwd=os.path.join(ROOT, "docs"))


build_cargo_docs()
build_doxygen_docs()


def setup(app):
    app.add_css_file("equistore.css")


# -- General configuration ---------------------------------------------------

needs_sphinx = "4.0.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "breathe",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["Thumbs.db", ".DS_Store"]


autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "short"

sphinx_gallery_conf = {
    "filename_pattern": "/*",
    "examples_dirs": ["../../python/examples"],
    "gallery_dirs": ["examples"],
    "min_reported_time": 60,
    # Make the code snippet for equistore functions clickable
    #"reference_url": {"equistore": None},
    #"prefer_full_module": ["equistore"],
}

breathe_projects = {
    "equistore": os.path.join(ROOT, "docs", "build", "doxygen", "xml"),
}
breathe_default_project = "equistore"
breathe_domain_by_extension = {
    "h": "c",
}

intersphinx_mapping = {
    "chemfiles": ("https://chemfiles.org/chemfiles.py/latest/", None),
    "rascaline": ("https://luthaf.fr/rascaline/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [os.path.join(ROOT, "docs", "static")]

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/lab-cosmo/equistore",
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
}

# font-awesome logos (used in the footer)
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
