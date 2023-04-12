import numpy as np

from equistore import Labels, TensorBlock, TensorMap


def drop_blocks(tensor: TensorMap, keys: Labels, copy: bool = False) -> TensorMap:
    """
    Drop specified key/block pairs from a TensorMap.

    :param tensor:
        the TensorMap to drop the key-block pair from.

    :param keys:
        a :py:class:`Labels` object containing the keys of the blocks to drop

    :param copy:
        if :py:obj:`True`, the returned :py:class:`TensorMap` is constructed by
        copying the blocks from the input `tensor`. If :py:obj:`False`
        (default), the values of the blocks in the output :py:class:`TensorMap`
        reference the same data as the input `tensor`. The latter can be useful
        for limiting memory usage, but should be used with caution when
        manipulating the underlying data.

    :return:
        the input :py:class:`TensorMap` with the specified key/block pairs
        dropped.
    """
    # Check arg types
    if not isinstance(tensor, TensorMap):
        raise TypeError(
            f"input `tensor` must be a TensorMap, got '{type(tensor)}' instead"
        )
    if not isinstance(keys, Labels):
        raise TypeError(
            f"input `keys` must be a Labels object, got '{type(keys)}' instead"
        )
    if not isinstance(copy, bool):
        raise TypeError(f"`copy` flag must be a boolean, got '{type(copy)}' instead")

    if not np.all(tensor.keys.names == keys.names):
        raise ValueError(
            "The input tensor's keys must have the same names as the specified"
            f" keys to drop. Should be {tensor.keys.names} but got {keys.names}"
        )

    # Find the difference between key of the original tensor and those to drop
    diff = np.setdiff1d(keys, tensor.keys)
    if len(diff) > 0:
        raise ValueError(
            "some keys in `keys` are not present in `tensor`."
            f" Non-existent keys: {diff}"
        )
    new_keys = np.setdiff1d(tensor.keys, keys)

    # Create the new TensorMap
    if copy:
        new_blocks = [tensor[key].copy() for key in new_keys]
    else:
        new_blocks = []
        for key in new_keys:
            # Create new block
            new_block = TensorBlock(
                values=tensor[key].values,
                samples=tensor[key].samples,
                components=tensor[key].components,
                properties=tensor[key].properties,
            )
            # Add gradients
            for parameter, gradient in tensor[key].gradients():
                new_block.add_gradient(
                    parameter=parameter,
                    data=gradient.data,
                    samples=gradient.samples,
                    components=gradient.components,
                )
            new_blocks.append(new_block)

    return TensorMap(keys=new_keys, blocks=new_blocks)
