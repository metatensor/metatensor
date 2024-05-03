# Pseudo-sphinx configuration to run sphinx-gallery as a command line tool

import os

from matplotlib.animation import Animation, FFMpegWriter, ImageMagickWriter
from sphinx_gallery.scrapers import figure_rst, matplotlib_scraper
from sphinx_gallery.sorting import FileNameSortKey


extensions = [
    "sphinx_gallery.gen_gallery",
]

HERE = os.path.dirname(__file__)
ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))


class AnimationScrapper:
    """
    Alternative image scrapper for sphinx-gallery, rendering the animation to GIF
    instead of a base64-encoded HTML video.

    This decreases the size of the documentation and the time taken to render the
    documentation (sphinx-gallery with ``matplotlib_animations=True``) renders the GIF
    and then re-render to create the HTML video.
    """

    def __init__(self):
        # only process animations once by storing their `id()` here
        self.seen = set()

    def __call__(self, block, block_vars, gallery_conf):
        animations = []
        for variable in block_vars["example_globals"].values():
            if isinstance(variable, Animation) and id(variable) not in self.seen:
                self.seen.add(id(variable))
                animations.append(variable)

        if len(animations) == 0:
            # just do the default scrapping
            return matplotlib_scraper(block, block_vars, gallery_conf)
        else:
            # process matplotlib static preview of the animation, and ignore it
            matplotlib_scraper(block, block_vars, gallery_conf)

            image_names = []
            for anim, image_path in zip(animations, block_vars["image_path_iterator"]):
                image_path = str(image_path)[:-4] + ".gif"
                image_names.append(image_path)

                # this is strongly inspired by the code in sphinx-gallery
                fig_size = anim._fig.get_size_inches()
                thumb_size = gallery_conf["thumbnail_size"]
                use_dpi = round(
                    min(t_s / f_s for t_s, f_s in zip(thumb_size, fig_size))
                )
                if FFMpegWriter.isAvailable():
                    writer = "ffmpeg"
                elif ImageMagickWriter.isAvailable():
                    writer = "imagemagick"
                else:
                    writer = None
                anim.save(str(image_path), writer=writer, dpi=use_dpi)

            return figure_rst(image_names, gallery_conf["src_dir"])


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
    "matplotlib_animations": False,
    "image_scrapers": (AnimationScrapper()),
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
