import docutils
from packaging.version import parse as version_parse
from sphinx.util.docutils import SphinxDirective


DOC_ROOT = "https://docs.metatensor.org/"


class VersionList(SphinxDirective):
    """
    Directive for a list of previous versions of the documentation.

    This is rendered by manually generating ReST using sphinx-design directives.
    """

    has_content = True
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        "tag-prefix": str,
        "url-suffix": str,
    }

    def run(self):
        # group together versions with the same major & minor components
        grouped_versions = {}
        content = self.parse_content_to_nodes()
        for node in content:
            if not isinstance(node, VersionNode):
                raise ValueError(
                    "only `.. version::` is allowed inside `.. version-list::`"
                )

            version = version_parse(node.version)
            group = (version.major, version.minor)
            if group not in grouped_versions:
                grouped_versions[group] = []

            grouped_versions[group].append(node)

        # generate ReST with the desired markup
        generated_content = """
.. grid::
    :margin: 0 0 0 0\n"""

        for group_i, (version_short, group) in enumerate(grouped_versions.items()):

            if group_i < 3:
                generated_content += f"""
    .. grid-item::
        :columns: 12 6 3 3
        :class: sd-p-1

        .. dropdown:: Version {version_short[0]}.{version_short[1]}
            :class-body: sd-mb-0  sd-pb-0
            :class-title: font-size-small\n"""
            elif group_i == 3:
                generated_content += """
    .. grid-item::
        :columns: 12 6 3 3
        :class: sd-p-1

        .. dropdown:: Older versions
            :class-body: sd-mb-0  sd-pb-0
            :class-title: font-size-small\n"""

            for node in group:
                version = node.version
                tag_prefix = self.options["tag-prefix"]

                url_suffix = (
                    node.url_suffix
                    if node.url_suffix is not None
                    else self.options["url-suffix"]
                )

                generated_content += f"""
            .. card:: {version}
                :class-body: sd-p-2
                :class-title: sd-mb-0
                :text-align: center
                :link: {DOC_ROOT}/{tag_prefix}{version}/{url_suffix}
                :link-type: url\n"""

        # parse the generated ReST
        return self.parse_text_to_nodes(generated_content)


class VersionNode(docutils.nodes.Node):
    """
    Node for a single version directive. This is only used to transmit information
    between the ``.. version::`` directive and the version list, and never rendered on
    its own.
    """

    def __init__(self, version, url_suffix):
        self.version = version
        self.url_suffix = url_suffix


class VersionItem(SphinxDirective):
    """
    A single item in a version list. This can override the url suffix if a different url
    was used for this version.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        "url-suffix": str,
    }

    def run(self):
        return [
            VersionNode(
                version=self.arguments[0],
                url_suffix=self.options.get("url-suffix"),
            )
        ]


def setup(app):
    app.add_directive("version-list", VersionList)
    app.add_directive("version", VersionItem)
