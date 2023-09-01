from typing import List, Optional, Union

from . import _dispatch
from ._classes import TensorBlock, TensorMap, check_isinstance, torch_jit_is_scripting


def to(
    tensor: TensorMap,
    backend: Optional[str] = None,
    dtype: Optional[_dispatch.torch_dtype] = None,
    device: Optional[Union[str, _dispatch.torch_device]] = None,
    requires_grad: Optional[bool] = None,
) -> TensorMap:
    """
    Converts a :py:class:`TensorMap` to a different backend. Currently only
    supports converting to and from numpy- or torch-based tensors.

    :param tensor: input :py:class:`TensorMap`.
    :param backend: :py:class:`str` indicating the backend to convert to.
        Currently only supports ``"numpy"`` or ``"torch"``. If not provided, the
        backend of the input ``tensor`` will be used.

    :param dtype: the dtype of the data in the resulting :py:class:`TensorMap`.
        This is passed directly to numpy/torch, so can be specified as a variety
        of objects, such as (but not limited to) :py:class:`numpy.dtype`,
        :py:class:`torch.dtype`, :py:class:`str`, or :py:class:`type`.
    :param device: only applicable if ``backend`` is ``"torch"``. The device on
        which the :py:class:`torch.Tensor` objects of the resulting
        :py:class:`TensorMap` should be stored. Can be specified as a variety of
        objects such as (but not limited to) :py:class:`torch.device` or
        :py:class:`str`.
    :param requires_grad: only applicable if ``backend`` is ``"torch"``. A
        :py:class:`bool` indicating whether or not to use torch's autograd to
        record operations on this block's data. If not specified (i.e.
        ``requires_grad=None``), in the case that the input ``tensor`` is
        already torch-based, the value of ``requires_grad`` will be preserved at
        its current setting. In the case that ``tensor`` is numpy-based, upon
        conversion to a torch tensor, torch will by default set
        ``requires_grad`` to ``False``.

    :return: a :py:class:`TensorMap` converted to the specified backend, data
        type, and/or device.
    """
    # Check types
    if torch_jit_is_scripting():
        is_tensor_map = isinstance(tensor, TensorMap)
    else:
        is_tensor_map = check_isinstance(tensor, TensorMap)
    if not is_tensor_map:
        raise TypeError("`tensor` should be a metatensor `TensorMap`")
    # Convert each block and build the return TensorMap
    keys = tensor.keys
    new_blocks = [
        block_to(
            tensor.block(keys.entry(i)).copy(),
            backend=backend,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        for i in range(keys.values.shape[0])
    ]

    return TensorMap(keys=keys, blocks=new_blocks)


def block_to(
    block: TensorBlock,
    backend: Optional[str] = None,
    dtype: Optional[_dispatch.torch_dtype] = None,
    device: Optional[Union[str, _dispatch.torch_device]] = None,
    requires_grad: Optional[bool] = None,
) -> TensorBlock:
    """
    Converts a :py:class:`TensorBlock` to a different ``backend``. Currently
    only supports converting to and from numpy- or torch-based tensors.

    :param block: input :py:class:`TensorBlock`.
    :param backend: :py:class:`str`, the backend to convert to. Currently only
        supports ``"numpy"`` or ``"torch"``. If not specified, the backend is
        set to match the current backend of the input ``block``.
    :param dtype: the dtype of the data in the resulting
        :py:class:`TensorBlock`. This is passed directly to numpy/torch, so can
        be specified as a variety of objects, such as (but not limited to)
        :py:class:`numpy.dtype`, :py:class:`torch.dtype`, :py:class:`str`, or
        :py:class:`type`.
    :param device: only applicable if ``backend`` is ``"torch"``. The device on
        which the :py:class:`torch.Tensor` of the resulting
        :py:class:`TensorBlock` should be stored. Can be specified as a variety
        of objects such as (but not limited to) :py:class:`torch.device` or
        :py:class:`str`.
    :param requires_grad: only applicable if ``backend`` is ``"torch"``. A
        :py:class:`bool` indicating whether or not to use torch's autograd to
        record operations on this block's data. If not specified (i.e.
        ``None``), in the case that the input ``block`` is already torch-based,
        the value of ``requires_grad`` will be preserved. In the case that
        ``block`` is numpy-based, upon conversion to a torch tensor, torch will
        by default set ``requires_grad`` to ``False``.

    :return: a :py:class:`TensorBlock` converted to the specified backend, data
        type, and/or device.
    """
    # Check inputs
    if torch_jit_is_scripting():
        is_tensor_block = isinstance(block, TensorBlock)
    else:
        is_tensor_block = check_isinstance(block, TensorBlock)
    if not is_tensor_block:
        raise TypeError("`block` should be a metatensor `TensorBlock`")
    if backend is not None:
        if not isinstance(backend, str):
            raise TypeError("'backend' should be given as a string")
        else:
            if backend not in ["numpy", "torch"]:
                raise ValueError(f"backend '{backend}' is not supported")
            if backend == "numpy":
                if requires_grad is not None:
                    raise ValueError(
                        "the `numpy` backend option does not support autograd"
                        " gradient tracking"
                    )

    # Walk the tree of gradients without recursion
    # (recursion is not supported by torchscript)

    # The current_location list of strings will contain the name of
    # all the gradients until the current location. This allows to
    # access parents of gradient blocks easily
    current_location: List[str] = []

    current_block = block  # the block that is being examined

    # transformed_blocks is a stack that will be populated and
    # depopulated during the algorithm
    transformed_blocks: List[TensorBlock] = []

    # last_visite keeps track of the last gradient block that has been
    # visited while walking backward. While walking forward, this variable
    # is an empty string
    last_visited = ""

    while True:
        gradient_names = current_block.gradients_list()
        n_gradients = len(gradient_names)
        if last_visited == "":
            # we're walking forward and it's the first time we see this block
            # transform and append to list of transformed blocks:
            transformed_blocks.append(
                _block_to(current_block, backend, dtype, device, requires_grad)
            )
            if n_gradients == 0:  # the current block has no gradients
                # step back:
                if len(current_location) == 0:
                    break  # algorithm completed
                last_visited = (
                    current_location.pop()
                )  # removes last visited gradient name and stores it
                current_block = _reach_current_block(
                    block, current_location
                )  # reach current location
            else:  # the current block has gradients
                # proceed walking forward:
                current_block = current_block.gradient(gradient_names[0])
                current_location.append(gradient_names[0])
        else:
            # we're walking back to a block we've already seen.
            # get index of the last gradient of the current block that
            # has been visited and converted:
            index_last_visited = gradient_names.index(last_visited)
            if (
                index_last_visited == n_gradients - 1
            ):  # the last visited gradient was the last one we needed to convert
                # add gradients blocks to the current block; these are the
                # last n_gradients blocks in transformed_blocks and the one before
                # them, respectively.
                for i_gradient in range(n_gradients):
                    transformed_blocks[-n_gradients - 1].add_gradient(
                        gradient_names[i_gradient],
                        transformed_blocks[i_gradient - n_gradients],
                    )
                # remove all added gradients from the transformed list:
                for _ in range(n_gradients):
                    transformed_blocks.pop()
                # the block and its gradients have been assembled. Step back:
                if len(current_location) == 0:
                    break  # algorithm completed
                last_visited = (
                    current_location.pop()
                )  # removes last visited gradient and stores it
                current_block = _reach_current_block(
                    block, current_location
                )  # reach current location
            else:  # more gradients to convert in the current block
                # walk forward:
                current_block = current_block.gradient(
                    gradient_names[index_last_visited + 1]
                )
                current_location.append(gradient_names[index_last_visited + 1])
                last_visited = ""  # walking forward

    # at this point, transformed_blocks only contains the final transformed block:
    return transformed_blocks[0]


def _block_to(
    block: TensorBlock,
    backend: Optional[str],
    dtype: Optional[_dispatch.torch_dtype] = None,
    device: Optional[Union[str, _dispatch.torch_device]] = None,
    requires_grad: Optional[bool] = None,
) -> TensorBlock:
    """
    Converts a :py:class:`TensorBlock`, but not its gradients, to a different
    ``backend``, dtype and/or device.
    """
    # Create new block, with the values tensor converted
    new_block = TensorBlock(
        values=_dispatch.to(
            array=block.values,
            backend=backend,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        ),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    return new_block


def _reach_current_block(block: TensorBlock, current_location: List[str]):
    # walks through the gradient path defined by current_location
    current_block = block
    for gradient_name in current_location:
        current_block = current_block.gradient(gradient_name)
    return current_block
