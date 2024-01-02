# Pseudo-sphinx configuration to run sphinx-gallery as a command line tool

import os


extensions = [
    "sphinx_gallery.gen_gallery",
]

HERE = os.path.dirname(__file__)
ROOT = os.path.realpath(os.path.join(HERE, "..", "..", ".."))


sphinx_gallery_conf = {
    "filename_pattern": ".*",
    "examples_dirs": [
        os.path.join(ROOT, "python", "examples", "atomistic"),
        os.path.join(ROOT, "python", "examples", "learn", "data"),
    ],
    "gallery_dirs": [
        os.path.join(ROOT, "docs", "src", "examples", "atomistic"),
        os.path.join(ROOT, "docs", "src", "examples", "learn", "data"),
    ],
    "matplotlib_animations": True,
    "default_thumb_file": os.path.join(
        ROOT, "docs", "static", "images", "TensorBlock-Basic.png"
    ),
}
