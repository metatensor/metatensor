from sphinx_design.grids import GridItemCardDirective


DOC_ROOT = "https://doc.metatensor.org/"


class GridItemVersion(GridItemCardDirective):
    """
    Wrapper for GridItemCardDirective (``.. grid-item-card::``) as used for versions
    lists in the docs.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        "tag-prefix": str,
        "url-suffix": str,
    }

    def run(self):
        version = self.arguments.pop()
        self.arguments.append(f"Version {version}")

        tag_prefix = self.options.pop("tag-prefix")
        url_suffix = self.options.pop("url-suffix")

        self.options["link"] = f"{DOC_ROOT}/{tag_prefix}{version}/{url_suffix}"
        self.options["link-type"] = "url"
        self.options["columns"] = [
            "sd-col-xs-12",
            "sd-col-sm-6",
            "sd-col-md-3",
            "sd-col-lg-3",
        ]
        # add padding all around the content
        self.options["class-body"] = ["sd-p-2"]
        # remove margin on the bottom
        self.options["class-title"] = ["sd-mb-0"]
        self.options["text-align"] = ["sd-text-center"]

        return super().run()


def setup(app):
    app.add_directive("grid-item-version", GridItemVersion)
