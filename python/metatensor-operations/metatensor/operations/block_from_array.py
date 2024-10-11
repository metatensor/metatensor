from typing import List, Optional

import numpy as np

from . import _dispatch
from ._backend import Labels, TensorBlock, torch_jit_is_scripting, torch_jit_script


try:
    import torch

    TorchScriptClass = torch.ScriptClass
except ImportError:

    class TorchScriptClass:
        pass


@torch_jit_script
def block_from_array(
    array,
    sample_names: Optional[List[str]] = None,
    component_names: Optional[List[str]] = None,
    property_names: Optional[List[str]] = None,
) -> TensorBlock:
    """
    Creates a simple TensorBlock from an array.

    The metadata in the resulting :py:class:`TensorBlock` is filled with ranges of
    integers. This function should be seen as a quick way of creating a
    :py:class:`TensorBlock` from arbitrary data. However, the metadata generated in this
    way has little meaning.

    :param array: An array with two or more dimensions. This can either be a
        :py:class:`numpy.ndarray` or a :py:class:`torch.Tensor`.
    :param sample_names: A list containing ``d_samples`` names for the sample
        dimensions. The first ``d_samples`` dimensions in the array will be interpreted
        as enumerating samples. ``None`` implies a single dimension named ``"sample"``.
    :param property_names: A list containing ``d_properties`` names for the property
        dimensions. The last ``d_properties`` dimensions in the array will be
        interpreted as enumerating properties. ``None`` implies a single dimension named
        ``"property"``.
    :param component_names: A list containing ``n_component`` names for the component
        dimensions. The middle ``d_components`` dimensions in the array will be
        interpreted as enumerating components. ``None`` implies that all the middle
        dimensions (after removing any sample and property dimensions) will be
        considered components, named ``"component_xxx"``.

    :return: A :py:class:`TensorBlock` whose values correspond to the provided
        ``array``. If no name options are provided, the metadata names are set to
        ``"sample"`` for samples; ``"component_1"``, ``"component_2"``, ... for
        components; and ``property`` for properties. The number of ``component`` labels
        is adapted to the dimensionality of the input array. If axes names are given, as
        indicated in the parameter list, the dimensions of the array will be interpreted
        accordingly, and indices also generated in a similar way. The metadata
        associated with each axis is a range of integers going from 0 to the size of the
        corresponding axis. The returned :py:class:`TensorBlock` has no gradients.

    >>> import numpy as np
    >>> import metatensor
    >>> # Construct a simple 4D array:
    >>> array = np.linspace(0, 10, 42).reshape((7, 3, 1, 2))
    >>> # Transform it into a TensorBlock:
    >>> tensor_block = metatensor.block_from_array(array)
    >>> print(tensor_block)
    TensorBlock
        samples (7): ['sample']
        components (3, 1): ['component_1', 'component_2']
        properties (2): ['property']
        gradients: None
    >>> # The data inside the TensorBlock will correspond to the provided array:
    >>> print(np.all(array == tensor_block.values))
    True
    >>> # High-dimensional tensor
    >>> array = np.linspace(0, 10, 60).reshape((2, 3, 5, 1, 2))
    >>> # Specify axes names:
    >>> tensor_block = metatensor.block_from_array(
    ...     array, sample_names=["a", "b"], property_names=["y"]
    ... )
    >>> print(tensor_block)
    TensorBlock
        samples (6): ['a', 'b']
        components (5, 1): ['component_1', 'component_2']
        properties (2): ['y']
        gradients: None
    """

    if torch_jit_is_scripting():
        # we are using metatensor-torch
        labels_array_like = torch.empty(0)
    else:
        if isinstance(Labels, TorchScriptClass):
            # we are using metatensor-torch
            labels_array_like = torch.empty(0)
        else:
            # we are using metatensor-core
            labels_array_like = np.empty(0)

    shape = array.shape
    n_dimensions = len(shape)
    if n_dimensions < 2:
        raise ValueError(
            f"the array provided to `block_from_array` \
            must have at least two dimensions. Too few provided: {n_dimensions}"
        )

    # constructs the default label names and counts
    if sample_names is None:
        sample_names = ["sample"]
    d_samples = len(sample_names)
    if property_names is None:
        property_names = ["property"]
    d_properties = len(property_names)
    # guess number of components
    d_components = n_dimensions - d_samples - d_properties
    if d_components < 0:
        raise ValueError(
            f"the array provided to `block_from_array` with shape {shape} "
            + "does not have enough dimensions to match the sample and property names"
        )
    if component_names is None:
        component_names = [
            f"component_{component_index + 1}"
            for component_index in range(d_components)
        ]
    if len(component_names) != d_components:
        raise ValueError(
            f"the array provided to `block_from_array` with shape {shape} "
            + "does not have enough dimensions to match the given sample, "
            + "component, and property names"
        )

    samples = Labels(
        names=sample_names,
        values=_dispatch.indices_like(shape[0:d_samples], labels_array_like),
    )
    components = [
        Labels(
            names=[component_names[component_index]],
            values=_dispatch.int_array_like(
                list(range(axis_size)), labels_array_like
            ).reshape(-1, 1),
        )
        for component_index, axis_size in enumerate(shape[d_samples:-d_properties])
    ]
    properties = Labels(
        names=property_names,
        values=_dispatch.indices_like(shape[-d_properties:], labels_array_like),
    )

    device = _dispatch.get_device(array)
    samples = samples.to(device)
    components = [component.to(device) for component in components]
    properties = properties.to(device)

    # reshape the array if multiple axes of the input array are grouped as samples or
    # properties (i.e. if `len(sample_names) > 1` or `len(property_names) > 1`)
    if d_samples > 1 or d_properties > 1:
        block_shape = [len(samples)]
        for i in range(d_samples, d_samples + d_components):
            block_shape.append(shape[i])
        block_shape.append(len(properties))
        array = array.reshape(block_shape)

    return TensorBlock(array, samples, components, properties)
