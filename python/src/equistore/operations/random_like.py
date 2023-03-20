from typing import List, Union

from equistore import TensorBlock, TensorMap

from . import _dispatch
from .equal_metadata import _check_parameters_in_gradient_block


def random_uniform_like(
    tensor: TensorMap, parameters: Union[List[str], str] = None
) -> TensorMap:
    """Return a new :py:class:`TensorMap` with the same metadata as tensor, and all
    values randomly sampled from the uniform distribution between 0 and 1.

    :param tensor: Input tensor from which the metadata is taken.
    :param parameters: Which gradient parameter to copy. If ``None`` (default)
                       all gradient of ``tensor`` are present in the new tensor.
                       If empty list ``[]`` no gradients information are copied.
    """
    blocks = []
    for block in tensor.blocks():
        blocks.append(random_uniform_like_block(block=block, parameters=parameters))
    return TensorMap(tensor.keys, blocks)


def random_uniform_like_block(
    block: TensorBlock, parameters: Union[List[str], str] = None
) -> TensorBlock:
    """Return a new :py:class:`TensorBlock` with the same metadata as block, and all
    values randomly sampled from the uniform distribution between 0 and 1.

    :param block: Input block from which the metadata is taken.
    :param parameters: Which gradient parameter to copy. If ``None`` (default)
                       all gradients of ``block`` are present in the new block.
                       If empty list ``[]`` no gradients information are copied.
    """
    values = _dispatch.random_uniform(block.values)
    result_block = TensorBlock(
        values=values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    if isinstance(parameters, str):
        parameters = [parameters]

    if parameters is None:
        parameters = block.gradients_list()
    else:
        _check_parameters_in_gradient_block(
            block=block, parameters=parameters, fname="random_uniform_like_block"
        )

    for parameter in parameters:
        gradient = block.gradient(parameter)
        gradient_data = _dispatch.random_uniform(gradient.data)

        result_block.add_gradient(
            parameter,
            gradient_data,
            gradient.samples,
            gradient.components,
        )

    return result_block
