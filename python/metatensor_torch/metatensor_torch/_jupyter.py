import html
from typing import Optional

import torch

from ._html import (
    TensorBlockData,
    block_html,
    labels_html,
    tensor_map_html,
)


def _is_script_object(value) -> (bool, str):
    if not isinstance(value, torch.ScriptObject):
        return False, ""

    try:
        script_type = value._type()
        qualified_name = script_type.qualified_name()
    except Exception:
        return False, ""

    return True, qualified_name


def _labels_repr_html(labels: torch.ScriptObject) -> str:
    values = labels.values.cpu().numpy()
    html = labels_html(
        labels.names,
        values,
        default_visible_entries=7,
        max_html_entries=200,
    )
    return (
        "<div><strong>metatensor.torch.Labels</strong></div>"
        + "<hr style='border: none; border-top: 1px solid #888; margin: 0;'>"
        + html
    )


def _labels_entry_repr_html(entry: torch.ScriptObject) -> str:
    values = entry.values.cpu().numpy()[None, :]
    html = labels_html(
        entry.names,
        values,
        default_visible_entries=1,
        max_html_entries=1,
    )
    return (
        "<div><strong>metatensor.torch.LabelsEntry</strong></div>"
        + "<hr style='border: none; border-top: 1px solid #888; margin: 0;'>"
        + html
    )


def _is_metatensor_tensor_block_script_object(value) -> bool:
    if not isinstance(value, torch.ScriptObject):
        return False

    try:
        script_type = value._type()
        qualified_name = script_type.qualified_name()
    except Exception:
        return False

    return qualified_name == "__torch__.torch.classes.metatensor.TensorBlock"


def _tensor_block_repr_html(block: torch.ScriptObject) -> str:
    def _block_data(blk):
        gradients = {}
        for parameter in blk.gradients_list():
            grad = blk.gradient(parameter)
            gradients[parameter] = _block_data(grad)

        return TensorBlockData(
            values_shape=tuple(blk.values.shape),
            samples=(blk.samples.names, blk.samples.values.cpu().numpy()),
            components=[(c.names, c.values.cpu().numpy()) for c in blk.components],
            properties=(
                blk.properties.names,
                blk.properties.values.cpu().numpy(),
            ),
            gradients=gradients,
        )

    data = _block_data(block)
    body = block_html(data, module="metatensor.torch")

    class_name = "metatensor.torch.TensorBlock"
    rest = f" with shape {tuple(block.values.shape)}"
    header = f"<div><strong>{class_name}</strong>{rest}</div>"
    hr = "<hr style='border: none; border-top: 1px solid #888; margin: 0;'>"

    return header + hr + body


def _tensor_map_repr_html(tensor_map: torch.ScriptObject) -> str:
    def _block_data(blk):
        gradients = {}
        for parameter in blk.gradients_list():
            grad = blk.gradient(parameter)
            gradients[parameter] = _block_data(grad)

        return TensorBlockData(
            values_shape=tuple(blk.values.shape),
            samples=(blk.samples.names, blk.samples.values.cpu().numpy()),
            components=[(c.names, c.values.cpu().numpy()) for c in blk.components],
            properties=(
                blk.properties.names,
                blk.properties.values.cpu().numpy(),
            ),
            gradients=gradients,
        )

    keys = tensor_map.keys
    blocks = [_block_data(tensor_map.block_by_id(i)) for i in range(len(keys))]

    body = tensor_map_html(
        keys.names,
        keys.values.cpu().numpy(),
        blocks,
        module="metatensor.torch",
    )

    n_blocks = len(keys)
    block_word = "block" if n_blocks == 1 else "blocks"
    header = (
        f"<div><strong>metatensor.torch.TensorMap</strong>"
        f" with {n_blocks} {block_word}</div>"
    )
    hr = "<hr style='border: none; border-top: 1px solid #888; margin: 0;'>"

    return header + hr + body


def _call_previous_formatter(previous, value) -> Optional[str]:
    if previous is None:
        return None

    try:
        return previous(value)
    except TypeError:
        try:
            return previous(value, None, False)
        except Exception:
            return None
    except Exception:
        return None


def _standard_html_fallback(value) -> str:
    return html.escape(repr(value))


def register_ipython_html_formatter() -> None:
    try:
        from IPython import get_ipython
    except Exception:
        return

    ipython = get_ipython()
    if ipython is None:
        return

    html_formatter = ipython.display_formatter.formatters.get("text/html")
    if html_formatter is None:
        return

    try:
        previous_formatter = html_formatter.lookup_by_type(torch.ScriptObject)
    except KeyError:
        previous_formatter = None

    def _script_object_html(value):
        is_script, qualified_name = _is_script_object(value)

        if is_script:
            if qualified_name == "__torch__.torch.classes.metatensor.Labels":
                return _labels_repr_html(value)

            if qualified_name == "__torch__.torch.classes.metatensor.LabelsEntry":
                return _labels_entry_repr_html(value)

            if _is_metatensor_tensor_block_script_object(value):
                return _tensor_block_repr_html(value)

            if qualified_name == "__torch__.torch.classes.metatensor.TensorMap":
                return _tensor_map_repr_html(value)

        previous = _call_previous_formatter(previous_formatter, value)
        if previous is not None:
            return previous

        return _standard_html_fallback(value)

    html_formatter.for_type(torch.ScriptObject, _script_object_html)
