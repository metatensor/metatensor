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
    Converts a :py:class:`TensorMap` or :py:class:`TensorBlock` to a different
    backend. Currently only supports converting to numpy or torch-based tensors.

    :param tensor: input :py:class:`TensorMap` or :py:class:`TensorBlock`.
    :param backend: :py:class:`str`, the backend to convert to. Currently only
        supports ``"numpy"`` or ``"torch"``.
    :param requires_grad: :py:class:`bool`, whether or not to use torch's
        autograd to record operations on this tensor. Only applicable if
        `backend` is set to ``"torch"``.
    :param dtype: :py:class:`torch.dtype`, the base dtype of the resulting torch
        tensor, only applicable if `backend` is set to ``"torch"``.
    :param device: :py:class:`torch.device`, the device on which the tensors of
        the resulting :py:class:`TensorMap` should be stored, only applicable if
        `backend` is set to ``"torch"``.

    :return: a :py:class:`TensorMap` or :py:class:`TensorBlock` converted to the
        specified backend.
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
    Converts a :py:class:`TensorMap` or :py:class:`TensorBlock` to a different
    backend. Currently only supports converting to numpy or torch-based tensors.

    :param tensor: input :py:class:`TensorMap` or :py:class:`TensorBlock`.
    :param backend: :py:class:`str`, the backend to convert to. Currently only
        supports ``"numpy"`` or ``"torch"``.
    :param requires_grad: :py:class:`bool`, whether or not to use torch's
        autograd to record operations on this tensor. Only applicable if
        `backend` is set to ``"torch"``.
    :param dtype: :py:class:`torch.dtype`, the base dtype of the resulting torch
        tensor, only applicable if `backend` is set to ``"torch"``.
    :param device: :py:class:`torch.device`, the device on which the tensors of
        the resulting :py:class:`TensorMap` should be stored, only applicable if
        `backend` is set to ``"torch"``.

    :return: a :py:class:`TensorMap` or :py:class:`TensorBlock` converted to the
        specified backend.
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
    Creates a new :py:class:`TensorBlock` where block values are PyTorch
    :py:class:`torch.tensor` objects. Assumes the block values are already as a
    type that is convertible to a :py:class:`torch.tensor`, such as a numpy
    array or RustNDArray. The resulting torch tensor dtypes are enforced as
    :py:class:`torch.float64`.

    :param block: input :py:class:`TensorBlock`, with block values as ndarrays.
    :param requires_grad: bool, whether or not to torch's autograd should record
        operations on this tensor.
    :param dtype: ``torch.dtype``, the base dtype of the resulting torch tensor,
        i.e. ``torch.float64``.
    :param device: ``torch.device``, the device on which the resulting torch
        tensor should be stored, i.e. ``torch.device("cpu")``.

    :return: a :py:class:`TensorBlock` whose values tensor is now of type
        :py:class:`torch.tensor`.
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
    Takes a TensorBlock object whose values are torch.tensor objects and
    converts them to numpy arrays of dtype np.float64. Returns a new TensorBlock
    object.
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
