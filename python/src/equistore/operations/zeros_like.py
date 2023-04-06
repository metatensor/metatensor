from typing import List, Union

from equistore import TensorBlock, TensorMap

from . import _dispatch
from .equal_metadata import _check_gradient_presence


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
    :param requires_grad: If autograd should record operations for the returned
                          tensor. This option is only relevant for torch.

    Here is an example using this function. First we create a ``TensorMap`` with
    just one block with two gradients, named ``alpha`` and ``beta``, containing
    random data for values and gradients.

    >>> import numpy as np
    >>> import equistore
    >>> from equistore import TensorBlock, TensorMap, Labels
    >>> np.random.seed(1)
    >>> block = TensorBlock(
    ...     values=np.random.rand(4, 3),
    ...     samples=Labels.arange("sample", 4),
    ...     components=[],
    ...     properties=Labels.arange("property", 3),
    ... )
    >>> block.add_gradient(
    ...     parameter="alpha",
    ...     data=np.random.rand(2, 3, 3),
    ...     samples=Labels(["sample", "atom"], np.array([[0, 0], [0, 2]])),
    ...     components=[Labels.arange("component", 3)],
    ... )
    >>> block.add_gradient(
    ...     parameter="beta",
    ...     data=np.random.rand(1, 3),
    ...     samples=Labels(["sample"], np.array([[0]])),
    ...     components=[],
    ... )
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    >>> tensor = TensorMap(keys, [block])
    >>> print(tensor.block(0))
    TensorBlock
        samples (4): ['sample']
        components (): []
        properties (3): ['property']
        gradients: ['alpha', 'beta']

    Here we use the function ``zeros_like`` to create a :py:class:`TensorMap`
    with the same metadata as ``tensor``, but with all values set equal to 0.

    >>> tensor_zeros = equistore.zeros_like(tensor)
    >>> print(tensor_zeros.block(0))
    TensorBlock
        samples (4): ['sample']
        components (): []
        properties (3): ['property']
        gradients: ['alpha', 'beta']
    >>> print(tensor_zeros.block(0).values)
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    >>> print(tensor_zeros.block(0).gradient("alpha").data)
    [[[0. 0. 0.]
      [0. 0. 0.]
      [0. 0. 0.]]
    <BLANKLINE>
     [[0. 0. 0.]
      [0. 0. 0.]
      [0. 0. 0.]]]

    Note that if we copy just the gradient ``alpha``, ``beta`` is no longer
    available.

    >>> tensor_zeros = equistore.zeros_like(tensor, parameters="alpha")
    >>> print(tensor_zeros.block(0).gradients_list())
    ['alpha']
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
        _check_gradient_presence(block=block, parameters=parameters, fname="zeros_like")

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
