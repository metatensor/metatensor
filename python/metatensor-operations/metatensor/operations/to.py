from typing import Optional

from . import _dispatch
from ._classes import TensorBlock, TensorMap


def to(
    tensor: TensorMap,
    backend: Optional[str] = None,
    dtype=None,
    device=None,
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
    if not isinstance(tensor, TensorMap):
        raise TypeError(
            f"`tensor` should be a metatensor `TensorMap`, got {type(tensor)}"
        )
    # Convert each block and build the return TensorMap
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
    if not isinstance(block, TensorBlock):
        raise TypeError(
            f"`block` should be a metatensor `TensorBlock`, got {type(block)}"
        )
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

    return _block_to(block, backend, dtype, device, requires_grad)


def _block_to(
    block: TensorBlock,
    backend: str,
    dtype=None,
    device=None,
    requires_grad: Optional[bool] = None,
) -> TensorBlock:
    """
    Converts a :py:class:`TensorBlock` and all its gradients to a different
    ``backend``.
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
    # Recursively convert all gradient blocks to numpy
    for parameter, gradient_block in block.gradients():
        new_gradient_block = _block_to(
            block=gradient_block,
            backend=backend,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        new_block.add_gradient(
            parameter,
            new_gradient_block,
        )

    return new_block
