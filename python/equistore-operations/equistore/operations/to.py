from typing import Optional, Union

import numpy as np

from equistore.core import TensorBlock, TensorMap

from . import _dispatch


def to(
    tensor: TensorMap,
    backend: Optional[str] = None,
    dtype=None,
    device=None,
    requires_grad: bool = False,
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
        :py:class:`TensorMap` should be stored. Can be specified as a variety
        of objects such as (but not limited to) :py:class:`torch.device` or
        :py:class:`str`.
    :param requires_grad: only applicable if ``backend`` is ``"torch"``. A
        :py:class:`bool` indicating whether or not to use torch's autograd to
        record operations on this block's data. Default is false.

    :return: a :py:class:`TensorMap` converted to the specified backend, data
        type, and/or device.
    """
    if not isinstance(tensor, TensorMap):
        raise TypeError(
            f"`tensor` should be an equistore `TensorMap`, got {type(tensor)}"
        )

    keys = tensor.keys
    new_blocks = [
        block_to(
            tensor[key].copy(),
            backend=backend,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        for key in keys
    ]

    return TensorMap(keys=keys, blocks=new_blocks)


def block_to(
    block: TensorBlock,
    backend: Optional[str] = None,
    dtype=None,
    device=None,
    requires_grad: bool = False,
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
        record operations on this block's data. Default is false.

    :return: a :py:class:`TensorBlock` converted to the specified backend, data
        type, and/or device.
    """

    if not isinstance(block, TensorBlock):
        raise TypeError(
            f"`block` should be an equistore `TensorBlock`, got {type(block)}"
        )

    if backend is None:  # infer the target backend from the current one
        if isinstance(block.values, np.ndarray):
            backend = "numpy"
        elif isinstance(block.values, torch.Tensor):
            backend = "torch"
        else:
            raise TypeError(
                f"detected unsupported backend {backend} in the provided block"
            )
    if not isinstance(backend, str):
        raise TypeError("`backend` should be passed as a `str`")

    if backend == "torch":
        return _to_torch_block(block, dtype, device, requires_grad)
    elif backend == "numpy":
        if requires_grad:
            raise ValueError(
                "the `numpy` backend option does not support autograd gradient tracking"
            )
        return _to_numpy_block(block, dtype)
    else:
        raise ValueError(f"backend ``{backend}`` not supported")


def _to_torch_block(
    block: TensorBlock,
    dtype,
    device,
    requires_grad: bool,
) -> TensorBlock:
    """
    Creates a new :py:class:`TensorBlock` where block values are
    :py:class:`torch.Tensor` objects.

    :param block: input :py:class:`TensorBlock`.
    :param dtype: the dtype of the data in the resulting
        :py:class:`TensorBlock`. This is passed directly to torch, so can be
        specified as a variety of objects, such as (but not limited to)
        :py:class:`torch.dtype`, :py:class:`str`, or :py:class:`type`.
    :param device: the device on which the :py:class:`torch.Tensor` of the
        resulting :py:class:`TensorBlock` should be stored. Can be specified as
        a variety of objects such as (but not limited to)
        :py:class:`torch.device` or :py:class:`str`.
    :param requires_grad: :py:class:`bool`, whether or not torch's autograd
        should record operations on this tensor.

    :return: a :py:class:`TensorBlock` whose values tensor is of type
        :py:class:`torch.Tensor`, according to the specified parameters.
    """

    # Create new block, with the values tensor converted to a torch tensor.
    new_block = TensorBlock(
        values=_dispatch.to(
            block.values,
            backend="torch",
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        ),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient_block in block.gradients():
        # Recursively convert all gradient blocks to torch
        new_gradient_block = _to_torch_block(
            gradient_block, dtype=dtype, device=device, requires_grad=requires_grad
        )
        new_block.add_gradient(
            parameter,
            new_gradient_block,
        )

    return new_block


def _to_numpy_block(block: TensorBlock, dtype) -> TensorBlock:
    """
    Converts a :py:class:`TensorBlock` object to a new :py:class:`TensorBlock`
    object whose ``values`` (and gradients) are a :py:class:`numpy.ndarray`,
    with the desired data type.

    :param block: input :py:class:`TensorBlock`.
    :param dtype: the dtype of the data in the resulting
        :py:class:`TensorBlock`. This is passed directly to numpy, so can be
        specified as a variety of objects, such as (but not limited to)
        :py:class:`numpy.dtype`, :py:class:`str`, or :py:class:`type`.

    :return: a :py:class:`TensorBlock` whose values array is a
        :py:class:`numpy.ndarray`, with the specified ``dtype``.
    """

    # Create new block, with the values tensor converted to a numpy array.
    new_block = TensorBlock(
        values=_dispatch.to(block.values, backend="numpy", dtype=dtype),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient_block in block.gradients():
        # Recursively convert all gradient blocks to numpy
        new_gradient_block = _to_numpy_block(gradient_block, dtype=dtype)
        new_block.add_gradient(
            parameter,
            new_gradient_block,
        )

    return new_block
