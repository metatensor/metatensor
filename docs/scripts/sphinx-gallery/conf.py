# Pseudo-sphinx configuration to run sphinx-gallery as a command line tool

import os


extensions = [
    "sphinx_gallery.gen_gallery",
]

HERE = os.path.dirname(__file__)
ROOT = os.path.realpath(os.path.join(HERE, "..", "..", ".."))


sphinx_gallery_conf = {
    "filename_pattern": ".*",
    "copyfile_regex": r".*\.(example|npz)",
    "examples_dirs": [
        os.path.join(ROOT, "python", "examples", "core"),
        os.path.join(ROOT, "python", "examples", "learn"),
        os.path.join(ROOT, "python", "examples", "atomistic"),
    ],
    "gallery_dirs": [
        os.path.join(ROOT, "docs", "src", "examples", "core"),
        os.path.join(ROOT, "docs", "src", "examples", "learn"),
        os.path.join(ROOT, "docs", "src", "examples", "atomistic"),
    ],
    "matplotlib_animations": True,
    "remove_config_comments": True,
    "default_thumb_file": os.path.join(
        ROOT, "docs", "static", "images", "TensorBlock-Basic.png"
    ),
}
