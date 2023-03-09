import numpy as np
import torch

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap


def to(
    tensor: Union[TensorMap, TensorBlock],
    backend: str,
    requires_grad: Optional[bool] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Union[TensorMap, TensorBlock]:
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
    if not isinstance(tensor, [TensorMap, TensorBlock]):
        raise TypeError(
            f"``tensor`` should be an equistore `TensorMap` or `TensorBlock`, got {type(tensor)}"
        )
    if not isinstance(backend, str):
        raise TypeError("`backend` should be passed as a `str`")
    if backend == "torch":
        # Check args
        if not isinstance(requires_grad, bool):
            raise TypeError("`requires_grad` should be passed as a `bool`")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("`dtype` should be passed as a `torch.dtype`")
        if not isinstance(device, torch.device):
            raise TypeError("`device` should be passed as a `torch.device`")
        # Convert tensor
        if isinstance(tensor, TensorMap):
            return _to_torch(tensor, requires_grad, dtype, device)
        # TensorBlock
        return _to_torch_block(tensor, **kwargs)
    elif backend == "numpy":  # numpy
        # Convert tensor
        if isinstance(tensor, TensorMap):
            return _to_numpy(tensor)
        # TensorBlock
        return _to_numpy_block(tensor)
    # `backend` is not "numpy" or "torch"
    raise ValueError(f"backend ``{backend}`` not supported")


def _to_torch(
    tensor: TensorMap,
    requires_grad: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> TensorMap:
    """
    Return a new :py:class:`TensorMap` where block values are
    :py:class:`torch.Tensor` objects.

    Whether autograd should be turned on, the dtype of the tensor, and the
    device can all be specified as arguments.

    Assumes the block values are already as a type that is convertible to a
    :py:class:`torch.tensor`, such as a numpy array or RustNDArray. The
    resulting torch tensor dtypes are enforced as :py:class:`torch.float64`.

    :param tensor: input :py:class:`TensorMap`, with block values as ndarrays.
    :param requires_grad: bool, whether or not to torch's autograd should record
        operations on this tensor.
    :param dtype: :py:class:`torch.dtype`, the base dtype of the resulting torch
        tensor, i.e. ``torch.float64``.
    :param device: :py:class:`torch.device``, the device on which the resulting
        torch tensor should be stored, i.e. ``torch.device("cpu")``.

    :return: a :py:class:`TensorMap` where the values tensors of each block are
        now of type :py:class:`torch.tensor`.
    """
    keys = tensor.keys
    torch_blocks = [
        _to_torch_block(
            tensor[key].copy(), requires_grad=requires_grad, dtype=dtype, device=device
        )
        for key in keys
    ]
    return TensorMap(keys=keys, blocks=torch_blocks)


def _to_torch_block(
    block: TensorBlock,
    requires_grad: bool,
    dtype: torch.dtype,
    device: torch.device,
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
    new_block = TensorBlock(
        values=torch.tensor(block.values, requires_grad=requires_grad, dtype=dtype).to(
            device.type
        ),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    # Add gradients to each block, again where the values tensor are torch
    # tensors.
    for parameter, gradient in block.gradients():
        new_block.add_gradient(
            parameter,
            torch.tensor(gradient.data, requires_grad=requires_grad, dtype=dtype).to(
                device.type
            ),
            gradient.samples,
            gradient.components,
        )
    return new_block


def _to_numpy(tensor: TensorMap) -> TensorMap:
    """
    Takes a TensorMap object whose block values are torch.tensor objects and
    converts them to numpy arrays of dtype np.float64. Returns a new TensorMap
    object.
    """
    keys = tensor.keys
    numpy_blocks = [_to_numpy_block(tensor[key].copy()) for key in keys]
    return TensorMap(keys=keys, blocks=numpy_blocks)


def _to_numpy_block(block: TensorBlock) -> TensorBlock:
    """
    Takes a TensorBlock object whose values are torch.tensor objects and
    converts them to numpy arrays of dtype np.float64. Returns a new TensorBlock
    object.
    """
    if isinstance(block.values, np.ndarray):
        return block.copy()

    # Create new block, with the values tensor converted to a torch tensor.
    new_block = TensorBlock(
        values=np.array(block.values.detach().numpy(), dtype=np.float64),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    # Add gradients to each block, again where the values tensor are torch
    # tensors.
    for parameter, gradient in block.gradients():
        new_block.add_gradient(
            parameter,
            np.array(gradient.data.detach().numpy(), dtype=np.float64),
            gradient.samples,
            gradient.components,
        )

    return new_block
