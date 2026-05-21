import html
from collections import namedtuple
from typing import List, Sequence


TensorBlockData = namedtuple(
    "TensorBlockData",
    ["values_shape", "samples", "components", "properties", "gradients"],
)
TensorBlockData.__doc__ = """Data needed to render a TensorBlock as HTML.

:ivar values_shape: shape of the block's values array
:ivar samples: ``(names, values)`` tuple for the samples labels
:ivar components: list of ``(names, values)`` tuples for the components labels
:ivar properties: ``(names, values)`` tuple for the properties labels
:ivar gradients: dict mapping gradient parameter names to ``TensorBlockData``
"""


_HEADER_STYLE = "background: aliceblue; text-align: center;"


# This function is used by both the core and torch implementations, do not change it in
# a way that would break backward compatibility without also bumping this package's
# version accordingly.
def labels_html(
    names: Sequence[str],
    values,
    *,
    default_visible_entries: int = 7,
    max_html_entries: int = 200,
) -> str:
    assert max_html_entries >= default_visible_entries
    n_rows = len(values)
    n_columns = max(1, len(names))
    if len(names) == 0:
        names = [""]

    visible_count = min(default_visible_entries, n_rows)
    hidden_total = n_rows - visible_count

    max_hidden_rows = max(0, max_html_entries - visible_count)
    hidden_show = min(hidden_total, max_hidden_rows)
    remaining_entries = hidden_total - hidden_show

    def _row_as_ints(index: int):
        return [int(value) for value in values[index]]

    widths = [len(name) + 2 for name in names]
    for i in range(min(n_rows, max_html_entries)):
        for j, value in enumerate(_row_as_ints(i)):
            widths[j] = max(widths[j], len(str(value)) + 2)

    visible_rows = [_row_as_ints(i) for i in range(visible_count)]
    hidden_rows = [
        _row_as_ints(i) for i in range(visible_count, visible_count + hidden_show)
    ]

    def _colgroup_html() -> str:
        result = ["<colgroup>"]
        for width in widths:
            result.append(f'<col style="width: {width}ch;">')
        result.append("</colgroup>")
        return "".join(result)

    def _tbody_html(rows) -> str:
        result = ["<tbody>"]
        for row in rows:
            result.append("<tr>")
            for value in row:
                result.append(f"<td style='text-align: center;'>{value}</td>")
            result.append("</tr>")
        result.append("</tbody>")
        return "".join(result)

    def _table_html(rows, include_header: bool) -> str:
        result = ["<table style='margin-top: 10px'>"]
        result.append(_colgroup_html())
        if include_header:
            result.append("<thead><tr>")
            for name in names:
                result.append(f"<th style='{_HEADER_STYLE}'>{html.escape(name)}</th>")
            result.append("</tr></thead>")
        result.append(_tbody_html(rows))
        result.append("</table>")
        return "".join(result)

    if hidden_total == 0:
        return _table_html(visible_rows, include_header=True)

    result = [_table_html(visible_rows, include_header=True)]
    if hidden_show == 0:
        # no room for any hidden rows in the <details>, just show a message
        result.append(f"<div>... and {hidden_total} more</div>")
        return "".join(result)

    result.append(f"<details><summary>Show {hidden_total} more entries</summary>")
    result.append("<table>")
    result.append(_colgroup_html())
    result.append(_tbody_html(hidden_rows))

    if remaining_entries > 0:
        result.append("<tbody>")
        result.append("<tr>")
        result.append(
            f'<td colspan="{n_columns}"'
            " style='text-align: left;'>"
            f"... and {remaining_entries} more</td>"
        )
        result.append("</tr>")
        result.append("</tbody>")

    result.append("</table>")

    result.append("</details>")
    return "".join(result)


def labels_html_horizontal(
    names: Sequence[str],
    values,
    *,
    default_visible_entries: int = 20,
    max_html_entries: int = 200,
) -> str:
    """Horizontal HTML representation of Labels for use inside TensorBlock.

    The first column contains the dimension names, and the remaining columns
    contain the entry values (up to ``default_visible_entries``). If there are
    more entries, a ``<details>`` element contains the full vertical
    :func:`labels_html` table.

    :param names: names of the dimensions
    :param values: 2D array of label values
    :param default_visible_entries: how many entries to show horizontally
    :param max_html_entries: maximum number of entries to render in the vertical
        fallback table
    """
    n_rows = len(values)

    if n_rows == 0:
        names = list(names) if len(names) > 0 else [""]
        result = ["<table style='margin-top: 4px;'><tbody>"]
        for name in names:
            result.append(
                f"<tr><th style='{_HEADER_STYLE}'>{html.escape(name)}</th>"
                "<td style='text-align: left;'>(empty)</td></tr>"
            )
        result.append("</tbody></table>")
        return "".join(result)

    visible_count = min(default_visible_entries, n_rows)
    hidden_total = n_rows - visible_count

    names = list(names) if len(names) > 0 else [""]
    n_dimensions = len(names)

    # collect visible values as ints, transposed (one row per dimension)
    visible_columns = []
    for i in range(visible_count):
        row = [int(v) for v in values[i]]
        visible_columns.append(row)

    result = ["<table style='margin-top: 4px;'><tbody>"]
    for d in range(n_dimensions):
        result.append("<tr>")
        result.append(f"<th style='{_HEADER_STYLE}'>{html.escape(names[d])}</th>")
        for col in visible_columns:
            result.append(f"<td style='text-align: center;'>{col[d]}</td>")
        if hidden_total > 0:
            result.append("<td style='text-align: center;'>...</td>")
        result.append("</tr>")
    result.append("</tbody></table>")

    if hidden_total > 0:
        result.append(
            f"<details style='margin-left: 1em;'>"
            f"<summary>Show all {n_rows} entries (vertical)</summary>"
        )
        result.append(
            labels_html(
                names,
                values,
                default_visible_entries=max_html_entries,
                max_html_entries=max_html_entries,
            )
        )
        result.append("</details>")

    return "".join(result)


# This function is used by both the core and torch implementations, do not change it in
# a way that would break backward compatibility without also bumping this package's
# version accordingly.
def block_html(block: TensorBlockData, *, module: str) -> str:
    """HTML representation for a TensorBlock body (without header).

    The header (class name + shape info) should be added by the caller, similar to how
    :func:`labels_html` is used by ``Labels._repr_html_``.

    :param block: :py:class:`TensorBlockData` describing the block
    :param module: module name to use in gradient summaries (e.g. ``"metatensor"`` or
        ``"metatensor.torch"``)
    """
    result = ["<div>"]
    samples_names, samples_values = block.samples
    result.append('<div style="margin-top: 4px;">')
    result.append("<div><strong>samples:</strong></div>")
    result.append(
        labels_html_horizontal(
            samples_names,
            samples_values,
            default_visible_entries=20,
            max_html_entries=200,
        )
    )
    result.append("</div>")

    component_names: List[str] = []
    for names, _ in block.components:
        component_names.extend(names)

    if len(component_names) != 0:
        result.append('<div style="margin-top: 4px;">')
        result.append("<div><strong>components:</strong></div>")
        for names, values in block.components:
            result.append(
                labels_html_horizontal(
                    names,
                    values,
                    default_visible_entries=20,
                    max_html_entries=200,
                )
            )
        result.append("</div>")

    properties_names, properties_values = block.properties
    result.append('<div style="margin-top: 4px;">')
    result.append("<div><strong>properties:</strong></div>")
    result.append(
        labels_html_horizontal(
            properties_names,
            properties_values,
            default_visible_entries=20,
            max_html_entries=200,
        )
    )
    result.append("</div>")

    if len(block.gradients) != 0:
        result.append('<div style="margin-top: 4px;">')
        result.append("<div><strong>gradients:</strong></div>")
        result.append(
            """<style>
            .mts-gradients tr:hover {
                background: transparent !important;
            }
            </style>"""
        )
        class_name = f"{module}.TensorBlock"
        result.append("<table class='mts-gradients' style='margin-top: 4px;'><tbody>")
        for parameter, grad_data in block.gradients.items():
            summary = f"<strong>{html.escape(class_name)}</strong>"
            summary += html.escape(f" with shape {tuple(grad_data.values_shape)}")
            result.append("<tr>")
            result.append(
                "<td style='vertical-align: middle; text-align: left; font-size:1.1em'>"
                f"{html.escape(parameter)}</td>"
            )
            result.append(
                "<td style='vertical-align: top; text-align: left;'>"
                f"<details><summary>{summary}</summary>"
                + "<div style='margin-top: 10px;'></div>"
                + block_html(grad_data, module=module)
                + "</details></td>"
            )
            result.append("</tr>")
        result.append("</tbody></table>")
        result.append("</div>")

    result.append("</div>")
    return "".join(result)


# This function is used by both the core and torch implementations, do not change it in
# a way that would break backward compatibility without also bumping this package's
# version accordingly.
def tensor_map_html(
    keys_names: Sequence[str],
    keys_values,
    blocks: Sequence[TensorBlockData],
    *,
    module: str,
    default_visible_entries: int = 7,
    max_html_entries: int = 20,
) -> str:
    """HTML representation for a TensorMap body (without header).

    Renders a table with one row per key, where the last column contains a
    ``<details>/<summary>`` with the block's HTML representation.

    :param keys_names: names of the key dimensions
    :param keys_values: 2D array of key values
    :param blocks: list of :py:class:`TensorBlockData`, one per key
    :param module: module name to use in block summaries
    :param default_visible_entries: how many key rows to show before truncating
    :param max_html_entries: maximum number of key rows to render in the
        hidden fallback table
    """
    assert max_html_entries >= default_visible_entries
    n_rows = len(keys_values)
    n_columns = max(1, len(keys_names))
    if len(keys_names) == 0:
        keys_names = [""]

    visible_count = min(default_visible_entries, n_rows)
    hidden_total = n_rows - visible_count

    max_hidden_rows = max(0, max_html_entries - visible_count)
    hidden_show = min(hidden_total, max_hidden_rows)
    remaining_entries = hidden_total - hidden_show

    def _row_as_ints(index: int):
        return [int(value) for value in keys_values[index]]

    visible_rows = [_row_as_ints(i) for i in range(visible_count)]
    hidden_rows = [
        _row_as_ints(i) for i in range(visible_count, visible_count + hidden_show)
    ]

    widths = [len(name) + 2 for name in keys_names]
    for i in range(min(n_rows, max_html_entries)):
        for j, value in enumerate(_row_as_ints(i)):
            widths[j] = max(widths[j], len(str(value)) + 2)

    def _colgroup_html() -> str:
        result = ["<colgroup>"]
        for width in widths:
            result.append(f'<col style="width: {width}ch;">')
        result.append("</colgroup>")
        return "".join(result)

    class_name = f"{module}.TensorBlock"

    def _block_cell(block_data: TensorBlockData) -> str:
        summary = f"<strong>{html.escape(class_name)}</strong>"
        summary += html.escape(f" with shape {tuple(block_data.values_shape)}")
        return (
            "<td style='vertical-align: top; text-align: left;'>"
            f"<details><summary>{summary}</summary>"
            + "<div style='margin-top: 10px;'></div>"
            + block_html(block_data, module=module)
            + "</details></td>"
        )

    def _table_html(rows, include_header: bool) -> str:
        result = [
            """<style>
            .mts-tensor-map tr:hover {
                background: transparent !important;
            }
            </style>""",
            "<table class='mts-tensor-map' style='margin-top: 10px;'>",
        ]
        result.append(_colgroup_html())
        if include_header:
            result.append("<thead><tr>")
            for name in keys_names:
                result.append(f"<th style='{_HEADER_STYLE}'>{html.escape(name)}</th>")
            result.append(f"<th style='{_HEADER_STYLE}'></th>")
            result.append("</tr></thead>")
        result.append("<tbody>")
        for i, row in enumerate(rows):
            result.append("<tr>")
            for value in row:
                result.append(f"<td style='text-align: center;'>{value}</td>")
            result.append(_block_cell(blocks[i]))
            result.append("</tr>")
        result.append("</tbody>")
        result.append("</table>")
        return "".join(result)

    if hidden_total == 0:
        return _table_html(visible_rows, include_header=True)

    result = [_table_html(visible_rows, include_header=True)]
    if hidden_show == 0:
        result.append(f"<div>... and {hidden_total} more</div>")
        return "".join(result)

    result.append(f"<details><summary>Show {hidden_total} more entries</summary>")
    result.append("<table class='mts-tensor-map'>")
    result.append(_colgroup_html())
    result.append("<tbody>")
    for i, row in enumerate(hidden_rows):
        result.append("<tr>")
        for value in row:
            result.append(f"<td style='text-align: center;'>{value}</td>")
        result.append(_block_cell(blocks[visible_count + i]))
        result.append("</tr>")
    result.append("</tbody>")
    result.append("</table>")

    if remaining_entries > 0:
        result.append("<tbody>")
        result.append("<tr>")
        result.append(
            f'<td colspan="{n_columns}"'
            " style='text-align: left;'>"
            f"... and {remaining_entries} more</td>"
        )
        result.append("</tr>")
        result.append("</tbody>")

    result.append("</details>")
    return "".join(result)
