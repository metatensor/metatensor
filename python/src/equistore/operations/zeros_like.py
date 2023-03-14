from typing import List, Union

from equistore import TensorBlock, Labels, TensorMap
import numpy as np
from . import _dispatch
from ._utils import _check_parameters_in_gradient_block


def zeros_like(
    tensor: TensorMap,
    parameters: Union[List[str], str] = None,
    requires_grad: bool = False,
) -> TensorMap:
    """Return a new :py:class:`TensorMap` with the same metadata as tensor,
    and all values equal to zero.

    :param tensor: Input tensor from which the metadata is taken.
    :param parameters: Which gradient parameter to copy. If ``None`` (default)
                       all gradient of ``tensor`` are present in the new tensor.
                       If empty list ``[]`` no gradients information are copied.
    :param requires_grad: If autograd should record operations for the returned tensor.
                          This option is only relevant for torch.

    >>> block = TensorBlock(
    ...     values=np.array([
    ...         [1, 2, 4],
    ...         [3, 5, 6],
    ...         [7, 8, 9],
    ...         [10, 11, 12],
    ...     ]),
    ...     samples=Labels(
    ...         ["structure", "center"],
    ...    np.array([
    ...         [0, 0],
    ...         [0, 1],
    ...         [1, 0],
    ...         [1, 1],
    ...    ]),
    ... ),
    ... components=[],
    ... properties=Labels(
    ...         ["properties"], np.array([[0], [1], [2]])
    ...         ),
    ...         )
    ...
    >>> block.add_gradient(
    ...    parameter="positions",
    ...    data=np.zeros((2, 3, 3)),
    ...    samples=Labels(
    ...        ["sample", "atom"],
    ...        np.array([
    ...            [0, 0],
    ...            [0, 2],
    ...        ]),
    ...    ),
    ...    components=[
    ...        Labels(["direction"], np.array([[0], [1], [2]])),
    ...    ]
    ... )
    ...
    >>> block.add_gradient(
    ...    parameter="alpha",
    ...    data=np.ones((2, 3, 3)),
    ...    samples=Labels(
    ...        ["sample", "atom"],
    ...        np.array([
    ...            [0, 0],
    ...            [0, 2],
    ...        ]),
    ...    ),
    ...    components=[
    ...        Labels(["direction"], np.array([[0], [1], [2]])),
    ...    ]
    ... )
    ...
    ...
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    ...
    >>> tensor = TensorMap(keys, [block])
    ...
    >>> print(tensor.block(0).gradient('alpha').data)
    [[[1. 1. 1.]
      [1. 1. 1.]
      [1. 1. 1.]]
    <BLANKLINE>
     [[1. 1. 1.]
      [1. 1. 1.]
      [1. 1. 1.]]]
    >>> tensor_zeros = zeros_like(tensor, parameters='alpha')
    ...
    >>> print(tensor_zeros.block(0).values)
    [[0 0 0]
     [0 0 0]
     [0 0 0]
     [0 0 0]]
    >>> print(tensor_zeros.block(0).gradient('alpha').data)
    [[[0. 0. 0.]
      [0. 0. 0.]
      [0. 0. 0.]]
    <BLANKLINE>
     [[0. 0. 0.]
      [0. 0. 0.]
      [0. 0. 0.]]]
    """

    blocks = []
    for block in tensor.blocks():
        blocks.append(
            zeros_like_block(
                block=block, parameters=parameters, requires_grad=requires_grad
            )
        )
    return TensorMap(tensor.keys, blocks)


def zeros_like_block(
    block: TensorBlock,
    parameters: Union[List[str], str] = None,
    requires_grad: bool = False,
) -> TensorBlock:
    """Return a new :py:class:`TensorBlock` with the same metadata as block,
    and all values equal to zero.

    :param block: Input block from which the metadata is taken.
    :param parameters: Which gradient parameter to copy. If ``None`` (default)
                       all gradient of ``tensor`` are present in the new tensor.
                       If empty list ``[]`` no gradients information are copied.
    :param requires_grad: If autograd should record operations for the returned tensor.
                          This option is only relevant for torch.
    """

    values = _dispatch.zeros_like(block.values, requires_grad=requires_grad)
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
            block=block, parameters=parameters, fname="zeros_like_block"
        )

    for parameter in parameters:
        gradient = block.gradient(parameter)
        gradient_data = _dispatch.zeros_like(gradient.data)

        result_block.add_gradient(
            parameter,
            gradient_data,
            gradient.samples,
            gradient.components,
        )

    return result_block
