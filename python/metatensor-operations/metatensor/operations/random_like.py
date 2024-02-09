from typing import List, Optional, Union

from . import _dispatch
from ._backend import TensorBlock, TensorMap, torch_jit_script
from ._utils import _check_gradient_presence_raise


@torch_jit_script
def random_uniform_like_block(
    block: TensorBlock,
    gradients: Optional[Union[List[str], str]] = None,
    requires_grad: bool = False,
) -> TensorBlock:
    """
    Return a new :py:class:`TensorBlock` with the same metadata as block, and
    all values randomly sampled from the uniform distribution between 0 and 1.

    :param block:
        Input block from which the metadata is taken.

    :param gradients:
        Which gradients should be present in the output. If this is
        :py:obj:`None` (default) all gradient of ``block`` are present in the
        new :py:class:`TensorBlock`. If this is an empty list ``[]``, no
        gradients information is copied.

    :param requires_grad:
        If autograd should record operations for the returned tensor. This
        option is only relevant for torch.
    """
    values = _dispatch.rand_like(
        block.values,
        requires_grad=requires_grad,
    )
    result_block = TensorBlock(
        values=values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    if isinstance(gradients, str):
        gradients = [gradients]

    if gradients is None:
        gradients = block.gradients_list()
    else:
        _check_gradient_presence_raise(
            block=block, parameters=gradients, fname="random_uniform_like"
        )

    for parameter in gradients:
        gradient = block.gradient(parameter)
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        gradient_values = _dispatch.rand_like(
            gradient.values,
            requires_grad=requires_grad,
        )

        result_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient_values,
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )

    return result_block


@torch_jit_script
def random_uniform_like(
    tensor: TensorMap,
    gradients: Optional[Union[List[str], str]] = None,
    requires_grad: bool = False,
) -> TensorMap:
    """
    Return a new :py:class:`TensorMap` with the same metadata as tensor, and all
    values randomly sampled from the uniform distribution between 0 and 1.

    :param tensor:
        Input tensor from which the metadata is taken.

    :param gradients:
        Which gradients should be present in the output. If this is
        :py:obj:`None` (default) all gradient of ``tensor`` are present in the
        new :py:class:`TensorMap`. If this is an empty list ``[]``, no gradients
        information is copied.

    :param requires_grad:
        If autograd should record operations for the returned tensor. This
        option is only relevant for torch.

    >>> import numpy as np
    >>> import metatensor
    >>> from metatensor import TensorBlock, TensorMap, Labels
    >>> np.random.seed(1)

    First we create a :py:class:`TensorMap` with just one block with two
    gradients, named ``alpha`` and ``beta``, containing random data:

    >>> block = TensorBlock(
    ...     values=np.random.rand(4, 3),
    ...     samples=Labels.range("sample", 4),
    ...     components=[],
    ...     properties=Labels.range("property", 3),
    ... )
    >>> block.add_gradient(
    ...     parameter="alpha",
    ...     gradient=TensorBlock(
    ...         values=np.random.rand(2, 3, 3),
    ...         samples=Labels(["sample", "atom"], np.array([[0, 0], [0, 2]])),
    ...         components=[Labels.range("component", 3)],
    ...         properties=block.properties,
    ...     ),
    ... )
    >>> block.add_gradient(
    ...     parameter="beta",
    ...     gradient=TensorBlock(
    ...         values=np.random.rand(1, 3),
    ...         samples=Labels(["sample"], np.array([[0]])),
    ...         components=[],
    ...         properties=block.properties,
    ...     ),
    ... )
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    >>> tensor = TensorMap(keys, [block])
    >>> print(tensor.block(0))
    TensorBlock
        samples (4): ['sample']
        components (): []
        properties (3): ['property']
        gradients: ['alpha', 'beta']

    Then we use ``random_uniform_like`` to create a :py:class:`TensorMap` with
    the same metadata as ``tensor``, but with all values randomly sampled from a
    uniform distribution.

    >>> tensor_random = metatensor.random_uniform_like(tensor)
    >>> print(tensor_random.block(0))
    TensorBlock
        samples (4): ['sample']
        components (): []
        properties (3): ['property']
        gradients: ['alpha', 'beta']
    >>> print(tensor_random.block(0).values)
    [[0.53316528 0.69187711 0.31551563]
     [0.68650093 0.83462567 0.01828828]
     [0.75014431 0.98886109 0.74816565]
     [0.28044399 0.78927933 0.10322601]]
    >>> print(tensor_random.block(0).gradient("alpha").values)
    [[[0.44789353 0.9085955  0.29361415]
      [0.28777534 0.13002857 0.01936696]
      [0.67883553 0.21162812 0.26554666]]
    <BLANKLINE>
     [[0.49157316 0.05336255 0.57411761]
      [0.14672857 0.58930554 0.69975836]
      [0.10233443 0.41405599 0.69440016]]]

    Note that if we copy just the gradient ``alpha``, ``beta`` is no longer
    available.

    >>> tensor_random = metatensor.random_uniform_like(tensor, gradients="alpha")
    >>> print(tensor_random.block(0).gradients_list())
    ['alpha']
    """
    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        blocks.append(
            random_uniform_like_block(
                block=block,
                gradients=gradients,
                requires_grad=requires_grad,
            )
        )
    return TensorMap(tensor.keys, blocks)
