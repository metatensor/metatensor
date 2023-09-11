from typing import List, Union

from . import _dispatch
from ._classes import Labels, TensorBlock, TensorMap


def _sort_single_block(
    block: TensorBlock,
    axes: List[str],
    descending: bool,
) -> TensorBlock:
    """
    Sorts a single TensorBlock without the user input checking and sorting of gradients
    """

    samples_names = block.samples.names
    samples_values = block.samples.values

    components_names: List[List[str]] = []
    components_values = []
    for component in block.components:
        components_names.append(component.names)
        components_values.append(component.values)

    properties_names = block.properties.names
    properties_values = block.properties.values

    values = block.values
    if "samples" in axes:
        sorted_idx = _dispatch.argsort_metadata_values(
            samples_values, reverse=descending
        )
        samples_values = samples_values[sorted_idx]
        values = values[sorted_idx]
    if "components" in axes:
        for i, _ in enumerate(block.components):
            sorted_idx = _dispatch.argsort_metadata_values(
                components_values[i], reverse=descending
            )
            components_values[i] = components_values[i][sorted_idx]
            values = _dispatch.take(values, sorted_idx, axis=i + 1)
    if "properties" in axes:
        sorted_idx = _dispatch.argsort_metadata_values(
            properties_values, reverse=descending
        )
        properties_values = properties_values[sorted_idx]
        values = _dispatch.take(values, sorted_idx, axis=-1)

    samples_labels = Labels(names=samples_names, values=samples_values)
    properties_labels = Labels(names=properties_names, values=properties_values)
    components_labels = [
        Labels(names=components_names[i], values=components_values[i])
        for i in range(len(components_names))
    ]

    return TensorBlock(
        values=values,
        samples=samples_labels,
        components=components_labels,
        properties=properties_labels,
    )


def sort_block(
    block: TensorBlock,
    axes: Union[str, List[str]] = "all",
    descending: bool = False,
) -> TensorBlock:
    """
    Rearanges the values of an block according to the order given by the sorted metadata
    of the given axes.

    This function creates copies of the metadata on the CPU to sort the metadata.  For
    that each row have to be interpreted as tuple which is not supported by torch.

    :param axes: axis of array to argsort :param stable: if true, the order of duplicate
    elements stays the same an the array :param descending: if false, the order is
    ascending :return: sorted tensor block

    >>> import numpy as np
    >>> import metatensor
    >>> from metatensor import TensorBlock, TensorMap, Labels
    >>> block = TensorBlock(
    ...     values=np.arange(9).reshape(3, 3),
    ...     samples=Labels(
    ...         ["structures", "centers"], np.array([[0, 3], [0, 1], [0, 2]])
    ...     ),
    ...     components=[],
    ...     properties=Labels(["n", "l"], np.array([[2, 0], [3, 0], [1, 0]])),
    ... )
    >>> print(block)
    TensorBlock
        samples (3): ['structures', 'centers']
        components (): []
        properties (3): ['n', 'l']
        gradients: None
    >>> block_sorted_stepwise = metatensor.sort_block(block, axes=["properties"])
    >>> # properties are are sorted which are columns in this case
    >>> block_sorted_stepwise.values
    array([[2, 0, 1],
           [5, 3, 4],
           [8, 6, 7]])
    >>> block_sorted_stepwise = metatensor.sort_block(
    ...     block_sorted_stepwise, axes=["samples"]
    ... )
    >>> # samples are are sorted which are rows in this case
    >>> block_sorted_stepwise.values
    array([[5, 3, 4],
           [8, 6, 7],
           [2, 0, 1]])
    >>> sorted_block = metatensor.sort_block(block)
    >>> np.all(
    ...     sorted_block.values == block_sorted_stepwise.values
    ... )  # doing everything at once and verify correctness
    True
    >>> # Example for sorting the gradient
    >>> sorted_block.add_gradient(
    ...     parameter="g",
    ...     gradient=TensorBlock(
    ...         values=np.arange(18).reshape(3, 2, 3),
    ...         samples=Labels(["sample"], np.array([[1], [2], [0]])),
    ...         components=[Labels.range("direction", 2)],
    ...         properties=sorted_block.properties,
    ...     ),
    ... )
    >>> sorted_grad_block = metatensor.sort_block(sorted_block)
    >>> sorted_grad_block.gradient("g").samples == Labels.range("sample", 3)
    True
    >>> sorted_grad_block.gradient("g").properties == sorted_block.properties
    True
    >>> sorted_grad_block.gradient(
    ...     "g"
    ... ).values  # sorting of components can be clearly seen
    array([[[12, 13, 14],
            [15, 16, 17]],
    <BLANKLINE>
           [[ 0,  1,  2],
            [ 3,  4,  5]],
    <BLANKLINE>
           [[ 6,  7,  8],
            [ 9, 10, 11]]])

    """
    if isinstance(axes, str):
        if axes == "all":
            axes_list = ["samples", "components", "properties"]
        elif axes not in ["samples", "components", "properties"]:
            raise ValueError(
                f"input parameter 'axes' may only be one of the strings 'samples', "
                f"components' or 'properties' but is '{axes}'"
            )
        else:
            axes_list = [axes]
    elif isinstance(axes, list):
        for axis in axes:
            if axis not in ["samples", "components", "properties"]:
                raise ValueError(
                    f"input parameter 'axes' may only contain the strings 'samples', "
                    f"components' or 'properties' but contains '{axis}'"
                )
        axes_list = axes
    else:
        raise TypeError("input paramater 'axes' should be of type str or list of str")

    result_block = _sort_single_block(block, axes_list, descending)

    # issue #68
    # for parameter, gradient in block.gradients().items():
    for parameter, gradient in block.gradients():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        result_block.add_gradient(
            parameter=parameter,
            gradient=_sort_single_block(gradient, axes_list, descending),
        )

    return result_block


def sort(
    tensor: TensorMap,
    axes: Union[str, List[str]] = "all",
    descending: bool = False,
) -> TensorMap:
    """
    Sorts each block separately, see :py:func:`sort_block` for more information

    This function creates copies of the metadata on the CPU to sort the metadata.
    For that each row have to be interpreted as tuple which is not supported by torch.

    :param axes: axis of array to argsort
    :param stable: if true, the order of duplicate elements stays the same an the array
    :param descending: if false, the order is ascending
    :return: sorted tensor map

    >>> import numpy as np
    >>> import metatensor
    >>> from metatensor import TensorBlock, TensorMap, Labels
    >>> block_1 = TensorBlock(
    ...     values=np.arange(9).reshape(3, 3),
    ...     samples=Labels(
    ...         ["structures", "centers"], np.array([[0, 3], [0, 1], [0, 2]])
    ...     ),
    ...     components=[],
    ...     properties=Labels(["n", "l"], np.array([[1, 0], [2, 0], [0, 0]])),
    ... )
    >>> block_2 = TensorBlock(
    ...     values=np.arange(3).reshape(1, 3),
    ...     samples=Labels(["structures", "centers"], np.array([[0, 0]])),
    ...     components=[],
    ...     properties=Labels(["n", "l"], np.array([[1, 0], [2, 0], [0, 0]])),
    ... )
    >>> tm = TensorMap(
    ...     keys=Labels(["species"], np.array([[1], [0]])), blocks=[block_2, block_1]
    ... )
    >>> metatensor.sort(tm)
    TensorMap with 2 blocks
    keys: species
             0
             1
    """
    blocks: List[TensorBlock] = []

    sorted_idx = _dispatch.argsort_metadata_values(
        tensor.keys.values, reverse=descending
    )
    for i in sorted_idx:
        blocks.append(
            sort_block(
                block=tensor.block(tensor.keys[int(i)]),
                axes=axes,
                descending=descending,
            )
        )
    keys = Labels(names=tensor.keys.names, values=tensor.keys.values[sorted_idx])
    return TensorMap(keys, blocks)
