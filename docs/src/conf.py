import os
import sys
import toml
import shutil
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.append(ROOT)

# -- Project information -----------------------------------------------------

project = "aml_storage"
copyright = "2022, Guillaume Fraux"
author = "Guillaume Fraux"


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


# -- General configuration ---------------------------------------------------

needs_sphinx = "4.0.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    # "sphinx_autodoc_typehints",
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


breathe_projects = {
    "aml_storage": os.path.join(ROOT, "docs", "build", "doxygen", "xml"),
}
breathe_default_project = "aml_storage"
breathe_domain_by_extension = {
    "h": "c",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
