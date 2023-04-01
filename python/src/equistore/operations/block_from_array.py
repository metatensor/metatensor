import equistore
from equistore import Labels, TensorBlock


def block_from_array(array: equistore.data.Array) -> TensorBlock:
    """Returns a simple TensorBlock from an array.

    The metadata in the resulting ``TensorBlock`` is filled with
    ranges of integers. This function should be seen as a quick
    way of creating a ``TensorBlock`` from arbitrary data. However,
    the metadata generated in this way has little meaning.

    :param array: An array with two or more dimensions. This can
        either be a `np.ndarray` or a `torch.Tensor`.

    :return: A ``TensorBlock`` whose values correspond to the
        provided ``array``. Its metadata consists of ``Labels`` whose
        names are ``sample``, ``component1``, ``component2``, ...,
        ``property``, in this order. The number of
        ``component`` labels is adapted to the dimensionality of the
        input array. The metadata associated with each label is a range
        of integers going from 0 to the dimensionality of the
        corresponding axis (not included). The output ``TensorBlock``
        has no gradients.

    Here is an example using this function:

    >>> import numpy as np
    >>> import equistore
    ...
    >>> # Construct a simple 3D array:
    >>> array = np.linspace(0, 10, 42).reshape((7, 3, 1, 2))
    >>> # Transform it into a TensorBlock:
    >>> tensor_block = equistore.block_from_array(array)
    >>> print(tensor_block)
    TensorBlock
        samples (7): ['sample']
        components (3, 1): ['component_1', 'component_2']
        properties (2): ['property']
        gradients: no
    >>> # The data inside the TensorBlock will correspond to the provided array:
    >>> print(np.allclose(array, tensor_block.values))
    True
    """

    shape = array.shape
    n_dimensions = len(shape)
    if n_dimensions < 2:
        raise ValueError(
            f"the array provided to `block_from_array` \
            must have at least two dimensions. Too few provided: {n_dimensions}"
        )

    components = [
        Labels.arange(f"component_{component_index+1}", axis_size)
        for component_index, axis_size in enumerate(shape[1:-1])
    ]

    return TensorBlock(
        values=array,
        samples=Labels.arange("sample", shape[0]),
        components=components,
        properties=Labels.arange("property", shape[-1]),
    )
