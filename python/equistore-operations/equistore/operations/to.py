import numpy as np


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from equistore.core import TensorBlock, TensorMap


def to(
    tensor,
    backend=None,
    dtype=None,
    device=None,
    requires_grad=False,
) -> TensorMap:
    """
    Converts a :py:class:`TensorMap` to a different backend. Currently only
    supports converting to numpy or torch-based tensors.

    :param tensor: input :py:class:`TensorMap`.
    :param backend: :py:class:`str`, the backend to convert to. Currently only
        supports ``"numpy"`` or ``"torch"``. If not provided, the backend of the
        input ``tensor`` will be used.
    :param dtype: :py:class:`numpy.dtype` or :py:class:`torch.dtype`, according
        to the desired ``backend``. It is the dtype of the data in the resulting
        :py:class:`TensorMap`.
    :param device: :py:class:`torch.device`, the device on which the tensors of
        the resulting :py:class:`TensorMap` should be stored, only applicable if
        `backend` is set to ``"torch"``.
    :param requires_grad: :py:class:`bool`, whether or not to use torch's
        autograd to record operations on this tensor. Only applicable if
        `backend` is set to ``"torch"``.

    :return: a :py:class:`TensorMap`` converted to the specified backend,
        data type, and/or device.
    """
    if not isinstance(tensor, TensorMap):
        raise TypeError(
            f"``tensor`` should be an equistore `TensorMap`, got {type(tensor)}"
        )

    keys = tensor.keys
    torch_blocks = [
        block_to(
            tensor[key].copy(),
            backend=backend,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        for key in keys
    ]

    return TensorMap(keys=keys, blocks=torch_blocks)


def block_to(
    block: TensorBlock,
    backend=None,
    dtype=None,
    device=None,
    requires_grad=False,
) -> TensorBlock:
    """
    Converts a :py:class:`TensorBlock` to a different backend. Currently only
    supports converting to numpy or torch-based tensors.

    :param tensor: input :py:class:`TensorBlock`.
    :param backend: :py:class:`str`, the backend to convert to. Currently only
        supports ``"numpy"`` or ``"torch"``. If not provided, the backend of the
        input ``block`` will be used.
    :param dtype: :py:class:`numpy.dtype` or :py:class:`torch.dtype`, according
        to the desired ``backend``. It is the dtype of the data in the resulting
        :py:class:`TensorBlock`.
    :param device: :py:class:`torch.device`, the device on which the data of
        the resulting :py:class:`TensorBlock` should be stored, only applicable if
        `backend` is set to ``"torch"``.
    :param requires_grad: :py:class:`bool`, whether or not to use torch's
        autograd to record operations on this block's data. Only applicable if
        `backend` is set to ``"torch"``.

    :return: a :py:class:`TensorBlock`` converted to the specified backend,
        data type, and/or device.
    """

    if not isinstance(block, TensorBlock):
        raise TypeError(
            f"``block`` should be an equistore `TensorBlock`, got {type(block)}"
        )

    if backend is None:
        if isinstance(block.values, np.ndarray):
            backend = "numpy"
        elif isinstance(block.values, torch.Tensor):
            backend = "torch"
        else:
            raise TypeError("detected unsupported backend in the provided block")
    if not isinstance(backend, str):
        raise TypeError("`backend` should be passed as a `str`")

    if backend == "torch":
        return _to_torch_block(block, dtype, device, requires_grad)
    elif backend == "numpy":
        if requires_grad:
            raise ValueError("The `numpy` backend option does not support gradients")
        return _to_numpy_block(block, dtype)
    else:
        raise ValueError(f"backend ``{backend}`` not supported")


def _to_torch_block(
    block,
    dtype,
    device,
    requires_grad,
) -> TensorBlock:
    """
    Creates a new :py:class:`TensorBlock` where block values are
    :py:class:`torch.tensor` objects.

    :param block: input :py:class:`TensorBlock`.
    :param dtype: ``torch.dtype``, the base dtype of the resulting torch tensor
    :param device: the device on which the resulting torch
        tensor should be stored.
    :param requires_grad: bool, whether or not torch's autograd should record
        operations on this tensor.

    :return: a :py:class:`TensorBlock` whose values tensor is of type
        :py:class:`torch.tensor`, according to the specified parameters.
    """

    # Create new block, with the values tensor converted to a torch tensor.

    if isinstance(block.values, np.ndarray):
        new_values = torch.tensor(
            block.values, dtype=dtype, device=device, requires_grad=requires_grad
        )
    elif isinstance(block.values, torch.Tensor):
        # we need this to keep gradients of the tensor
        new_values = block.values.to(dtype=dtype, device=device)
    else:
        raise ValueError("backend not supported")

    new_block = TensorBlock(
        values=new_values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient_block in block.gradients():
        new_gradient_block = _to_torch_block(
            gradient_block, dtype=dtype, device=device, requires_grad=requires_grad
        )
        new_block.add_gradient(
            parameter,
            new_gradient_block,
        )

    return new_block


def _to_numpy_block(block, dtype) -> TensorBlock:
    """
    Converts a :py:class:`TensorBlock` object to a new :py:class:`TensorBlock`
    object whose ``values`` (and gradient values) are a :py:class:`numpy.ndarray`,
    with the desired data type.

    :param block: input :py:class:`TensorBlock`.
    :param dtype: ``numpy.dtype``, the base dtype of the resulting block.

    :return: a :py:class:`TensorBlock` whose ``values`` array is of type
        :py:class:`numpy.ndarray`, with the specified ``dtype``.
    """

    # Create new block, with the values tensor converted to a numpy array.
    new_block = TensorBlock(
        values=np.array(block.values, dtype=dtype),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient_block in block.gradients():
        new_gradient_block = _to_numpy_block(gradient_block, dtype=dtype)
        new_block.add_gradient(
            parameter,
            new_gradient_block,
        )

    return new_block
