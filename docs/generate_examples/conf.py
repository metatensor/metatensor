# Pseudo-sphinx configuration to run sphinx-gallery as a command line tool

import os

from chemiscope.sphinx import ChemiscopeScraper
from sphinx_gallery.sorting import FileNameSortKey


extensions = [
    "sphinx_gallery.gen_gallery",
]

HERE = os.path.dirname(__file__)
ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))


sphinx_gallery_conf = {
    "filename_pattern": ".*",
    "copyfile_regex": r".*\.(example|mts|xyz)",
    "examples_dirs": [
        os.path.join(ROOT, "python", "examples", "core"),
        os.path.join(ROOT, "python", "examples", "learn"),
    ],
    "gallery_dirs": [
        os.path.join(ROOT, "docs", "src", "examples", "core"),
        os.path.join(ROOT, "docs", "src", "examples", "learn"),
    ],
    "matplotlib_animations": False,
    "image_scrapers": ("matplotlib", ChemiscopeScraper()),
    "remove_config_comments": True,
    "within_subsection_order": FileNameSortKey,
    "default_thumb_file": os.path.join(
        ROOT, "docs", "static", "images", "TensorBlock-Basic.png"
    ),
    "reference_url": {"metatensor": None},
    "prefer_full_module": [
        "metatensor",
        r"metatensor\.learn\.data",
    ],
}
