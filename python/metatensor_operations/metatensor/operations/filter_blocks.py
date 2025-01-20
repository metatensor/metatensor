from typing import List

from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    is_metatensor_class,
    torch_jit_is_scripting,
    torch_jit_script,
)


@torch_jit_script
def filter_blocks(tensor: TensorMap, keys: Labels, copy: bool = False) -> TensorMap:
    """
    Returns a :py:class:`TensorMap` with the blocks filtered according to the
    :py:class:`Labels` key selection passed in ``keys``.

    :param tensor: the :py:class:`TensorMap` from which the keys will be selected.
    :param keys: the :py:class:`Labels` containing the keys indexing blocks to
        retain from the tensor :py:class:`TensorMap`.

    :param copy: if :py:obj:`True`, the returned :py:class:`TensorMap` is constructed by
        copying the blocks from the input `tensor`. If :py:obj:`False` (default), the
        values of the blocks in the output :py:class:`TensorMap` reference the same data
        as the input `tensor`. The latter can be useful for limiting memory usage, but
        should be used with caution when manipulating the underlying data.
    """
    # Check arg types
    if not torch_jit_is_scripting():
        if not is_metatensor_class(tensor, TensorMap):
            raise TypeError(
                f"`tensor` must be a metatensor TensorMap, not {type(tensor)}"
            )

        if not is_metatensor_class(keys, Labels):
            raise TypeError(f"`keys` must be a metatensor Labels, not {type(keys)}")

        if not isinstance(copy, bool):
            raise TypeError(f"`copy` must be a boolean, not {type(copy)}")

    tensor_keys = tensor.keys
    to_keep_indices: List[int] = tensor_keys.select(keys).tolist()

    # Create the new TensorMap
    new_blocks: List[TensorBlock] = []
    new_keys_values = []
    for i in range(len(tensor_keys)):
        if i not in to_keep_indices:
            continue

        new_keys_values.append(tensor_keys.entry(i).values)
        block = tensor[i]

        if copy:
            new_blocks.append(block.copy())
        else:
            # just increase the reference count on everything
            new_block = TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )

            for parameter, gradient in block.gradients():
                if len(gradient.gradients_list()) != 0:
                    raise NotImplementedError(
                        "gradients of gradients are not supported"
                    )

                new_block.add_gradient(
                    parameter=parameter,
                    gradient=TensorBlock(
                        values=gradient.values,
                        samples=gradient.samples,
                        components=gradient.components,
                        properties=new_block.properties,
                    ),
                )

            new_blocks.append(new_block)

    if len(new_keys_values) != 0:
        new_keys = Labels(tensor_keys.names, _dispatch.stack(new_keys_values, 0))
    else:
        new_keys = Labels(
            names=tensor_keys.names,
            values=_dispatch.empty_like(
                tensor_keys.values, (0, len(tensor_keys.names))
            ),
        )

    return TensorMap(keys=new_keys, blocks=new_blocks)
