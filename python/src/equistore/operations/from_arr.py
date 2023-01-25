import numpy as np

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap


def matrix_to_block(a: np.ndarray) -> TensorBlock:
    """Create a :class:`equistore.TensorBlock` from a 2d :class`numpy.ndarray`.

    The values of the block are the same as ``a``. The name of the properties label
    is ``'property'`` and name of the samples label is ``'sample'``. The block has
    no components.

    :param a: 2d numpy array used for the blocks values
    :returns block: block with filled values

    Example:
    >>> a = np.zeros([2,2])
    >>> block = matrix_to_block(a)
    >>> print(block)
    """

    if len(a.shape) != 2:
        raise ValueError(f"`a` has {len(a.shape)} but must have exactly 2")

    n_samples, n_properties = a.shape

    samples = Labels(["sample"], np.arange(n_samples).reshape(-1, 1))
    properties = Labels(["sample"], np.arange(n_properties).reshape(-1, 1))

    block = TensorBlock(
        values=a,
        samples=samples,
        components=[],
        properties=properties,
    )

    return block


def tensor_to_tensormap(a: np.ndarray) -> TensorMap:
    """Create a :class:`equistore.TensorMap` from 3d :class`numpy.ndarray`.

    First dimension of ``a`` defines the number of blocks created. The name of the
    keys label of the TensorMap is ``'keys'``.
    The values of each block are taken from the second and the third dimension of ``a``.
    The name of the properties label in each block is ``'property``. The name of the
    samples label in each block is ``'sample'``. The blocks have no components.

    :param a: 3d numpy array for the block of the TensorMap values
    :returns: TensorMap with filled values

    Example:
    >>> a = np.zeros([2,2])
    >>> # make 2d array 3d tensor
    >>> tensor = tensor_to_tensormap(a[np.newaxis, :])
    >>> print(tensor)
    """
    if len(a.shape) != 3:
        raise ValueError(f"`a` has {len(a.shape)} but must have exactly 3")

    blocks = []
    for values in a:
        blocks.append(matrix_to_block(values))

    keys = Labels(["keys"], np.arange(len(blocks)).reshape(-1, 1))
    return TensorMap(keys, blocks)
