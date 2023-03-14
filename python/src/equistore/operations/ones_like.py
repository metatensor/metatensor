from typing import List, Union
import numpy as np

from equistore import TensorBlock, TensorMap, Labels

from . import _dispatch
from ._utils import _check_parameters_in_gradient_block


def ones_like(
    tensor: TensorMap,
    parameters: Union[List[str], str] = None,
    requires_grad: bool = False,
) -> TensorMap:
    """Return a new :py:class:`TensorMap` with the same metadata as tensor,
    and all values equal to one.

    :param tensor: Input tensor from which the metadata is taken.
    :param parameters: Which gradient parameter to copy. If ``None`` (default)
                       all gradient of ``tensor`` are present in the new tensor.
                       If empty list ``[]`` no gradients information are copied.
    :param requires_grad: If autograd should record operations for the returned tensor.
                          This option is only relevant for torch.

    Here is an example using this function:

    First we create a ``TensorMap`` with just one block with two gradients, named ``alpha`` and `beta`.

    >>> block = TensorBlock(
    ...     values=np.array([
    ...         [1, 2, 4],
    ...         [3, 5, 6],
    ...         [7, 8, 9],
    ...         [10, 11, 12],
    ...     ]),
    ...     samples=Labels(
    ...         ["structure", "center"],
    ...         np.array([
    ...             [0, 0],
    ...             [0, 1],
    ...             [1, 0],
    ...             [1, 1],
    ...             ]),
    ...     ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["properties"], np.array([[0], [1], [2]])
    ...     ),
    ... )
    ...
    >>> np.random.seed(1)
    ...
    >>> block.add_gradient(
    ...    parameter="alpha",
    ...    data=np.random.rand(2, 3, 3),
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
    ...    parameter="beta",
    ...    data=np.random.rand(2, 3, 3),
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
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    ...
    >>> tensor = TensorMap(keys, [block])
    ...
    >>> print(tensor.block(0))
    TensorBlock
        samples (4): ['structure', 'center']
        components (): []
        properties (3): ['properties']
        gradients: ['alpha', 'beta']
    >>> print(tensor.block(0).gradient('alpha').data)
    [[[4.17022005e-01 7.20324493e-01 1.14374817e-04]
      [3.02332573e-01 1.46755891e-01 9.23385948e-02]
      [1.86260211e-01 3.45560727e-01 3.96767474e-01]]
    <BLANKLINE>
     [[5.38816734e-01 4.19194514e-01 6.85219500e-01]
      [2.04452250e-01 8.78117436e-01 2.73875932e-02]
      [6.70467510e-01 4.17304802e-01 5.58689828e-01]]]

    Here we use the function ``one_like`` to create one ``TensorMap`` with the same metadata as tensor,
    but with all values set equal to 1.

    >>> tensor_ones = ones_like(tensor)
    ...
    >>> print(tensor_ones.block(0))
    TensorBlock
        samples (4): ['structure', 'center']
        components (): []
        properties (3): ['properties']
        gradients: ['alpha', 'beta']
    >>> print(tensor_ones.block(0).values)
    [[1 1 1]
     [1 1 1]
     [1 1 1]
     [1 1 1]]
    >>> print(tensor_ones.block(0).gradient('alpha').data)
    [[[1. 1. 1.]
      [1. 1. 1.]
      [1. 1. 1.]]
    <BLANKLINE>
     [[1. 1. 1.]
      [1. 1. 1.]
      [1. 1. 1.]]]

    Note that if we copy just the gradient ``alpha``, ``beta`` is no more available.

    >>> tensor_ones = ones_like(tensor,  parameters='alpha')
    ...
    >>> print(tensor_ones.block(0))
    TensorBlock
        samples (4): ['structure', 'center']
        components (): []
        properties (3): ['properties']
        gradients: ['alpha']
    """

    blocks = []
    for block in tensor.blocks():
        blocks.append(
            ones_like_block(
                block=block, parameters=parameters, requires_grad=requires_grad
            )
        )
    return TensorMap(tensor.keys, blocks)


def ones_like_block(
    block: TensorBlock,
    parameters: Union[List[str], str] = None,
    requires_grad: bool = False,
) -> TensorBlock:
    """Return a new :py:class:`TensorBlock` with the same metadata as block,
    and all values equal to one.

    :param block: Input block from which the metadata is taken.
    :param parameters: Which gradient parameter to copy. If ``None`` (default)
                       all gradient of ``tensor`` are present in the new tensor.
                       If empty list ``[]`` no gradients information are copied.
    :param requires_grad: If autograd should record operations for the returned tensor.
                          This option is only relevant for torch.


    
    """

    values = _dispatch.ones_like(block.values, requires_grad=requires_grad)
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
            block=block, parameters=parameters, fname="ones_like_block"
        )

    for parameter in parameters:
        gradient = block.gradient(parameter)
        gradient_data = _dispatch.ones_like(gradient.data)

        result_block.add_gradient(
            parameter,
            gradient_data,
            gradient.samples,
            gradient.components,
        )

    return result_block
