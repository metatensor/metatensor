import numpy as np

from equistore import Labels, TensorMap


def drop_blocks(tensor: TensorMap, keys: Labels) -> TensorMap:
    """
    Drop specified key/block pairs from a TensorMap.

    :param tensor: the TensorMap to drop the key-block pair from.
    :param keys: a :py:class:`Labels` object containing the keys of the
        blocks to drop

    :return: the input :py:class:`TensorMap` with the specified key/block
        pairs dropped.
    """
    if not np.all(tensor.keys.names == keys.names):
        raise ValueError(
            "The input tensor's keys must have the same names as the specified"
            + f" keys to drop. Should be {tensor.keys.names} but got {keys.names}"
        )
    for key in keys:
        if not (key in tensor.keys):
            raise ValueError(f"{key} key does not exist in {tensor}")
    new_keys = np.setdiff1d(tensor.keys, keys)
    new_blocks = [tensor[key].copy() for key in new_keys]
    return TensorMap(keys=new_keys, blocks=new_blocks)
