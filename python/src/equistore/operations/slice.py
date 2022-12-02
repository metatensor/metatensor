import warnings

import numpy as np

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap


def slice(tensor: TensorMap, samples_to_slice=None, properties_to_slice=None):
    """
    Slices an input :py:class:`TensorMap` along the samples and/or properties
    dimension(s). ``samples_to_slice`` and ``properties_to_slice`` are
    :py:class:`Labels` objects that specify the samples/properties
    (respectively) names and indices that should be sliced, i.e. kept in the
    output tensor.

    Note that either ``samples_to_slice`` or ``properties_to_slice``, or both,
    should be specified as input.

    .. code-block:: python

        sliced_tensor = slice(
            tensor,
            samples=Labels(
                names=["structure", "center"],
                values=np.array(
                    [[0, 1], [0, 6], [1, 6], [3, 16]]
                ),  # must be a 2D-array
            )
            properties=Labels(
                names=["n",],  # radial channel
                values=np.array([[3,], [4,], [5,]]),
            )
        )

    Also note that this function will return a :py:class:`TensorMap` whose
    blocks are of equal or smaller dimensions (due to slicing) than those of the
    input. However, the returned :py:class:`TensorMap` will be returned with the
    same number of blocks and the corresponding keys as the input. If any block
    upon slicing is reduced to nothing, i.e. in the case that it has none of the
    specified ``samples_to_slice`` or ``properties_to_slice``, an empty block
    will be returned but will still be accessible by its key. User warnings will
    be issued if any blocks are sliced to contain no values.

    For the empty blocks that may be returned, although there will be no actual
    values in its ``TensorBlock.values`` array, the shape of this array will be
    non-zero in the dimensions that haven't been sliced. This allows the slicing
    of dimensions to be tracked.

    For example, if a TensorBlock of shape (52, 1, 5) is passed, and only some
    samples are specified to be sliced but none of these appear in the input
    TensorBlock, the returned TensorBlock values array will be empty, but its
    shape will be (0, 1, 5) - i.e. the samples dimension has been sliced to zero
    but the components and properties dimensions remain in-tact. The same logic
    applies to any Gradient TensorBlocks the input TensorBlock may have
    associated with it.

    See the documentation for the :py:func:`slice_block` function to see how an
    individual :py:class:`TensorBlock` is sliced.

    :param tensor: the input :py:class:`TensorMap` to be sliced.
    :param samples_to_slice: a :py:class:`Labels` object containing the names
        and indices of samples to keep in the each of the sliced
        :py:class:`TensorBlock` of the output :py:class:`TensorMap`. Default
        value of None indicates no slicing along the samples dimension should
        occur.
    :param properties_to_slice: a :py:class:`Labels` object containing the names
        and indices of properties to keep in each of the sliced
        :py:class:`TensorBlock` of the output :py:class:`TensorMap`. Default
        value of None indicates no slicing along the properties dimension should
        occur.

    :return: a :py:class:`TensorMap` that corresponds to the sliced input
        tensor.
    """

    # Perform input checks
    if not isinstance(tensor, TensorMap):
        raise TypeError(
            "the input tensor must be a `TensorMap` object, if you want to "
            "to slice a `TensorBlock`, use `slice_block()` instead"
        )

    _check_slice_types(
        samples_to_slice=samples_to_slice,
        properties_to_slice=properties_to_slice,
        sample_names=tensor.sample_names,
        property_names=tensor.property_names,
    )

    # Slice TensorMap
    sliced_tensor = _slice_tensormap(
        tensor,
        samples_to_slice=samples_to_slice,
        properties_to_slice=properties_to_slice,
    )

    # Calculate which blocks have been sliced to now be empty. True if any
    # dimension of block.values is 0, False otherwise.
    empty_blocks = np.array(
        [np.any(np.array(block.values.shape) == 0) for _, block in sliced_tensor]
    )

    # Issue warnings if some or all of the blocks are now empty.
    if np.any(empty_blocks):
        if np.all(empty_blocks):
            warnings.warn(
                "All TensorBlocks in the sliced TensorMap are now empty, "
                "based on your choice of samples and/or properties to slice by. "
            )
        else:
            warnings.warn(
                "Some TensorBlocks in the sliced TensorMap are now empty, "
                "based on your choice of samples and/or properties to slice by. "
                "The keys of the empty TensorBlocks are:\n "
                f"{tensor.keys[empty_blocks]}"
            )

    return sliced_tensor


def slice_block(
    block: TensorBlock,
    samples_to_slice=None,
    properties_to_slice=None,
) -> TensorBlock:
    """
    Slices an input :py:class:`TensorBlock` along the samples and/or properties
    dimension(s). ``samples_to_slice`` and ``properties_to_slice`` are
    :py:class:`Labels` objects that specify the samples/properties
    (respectively) names and indices that should be sliced, i.e. kept in the
    output :py:class:`TensorBlock`.

    Note that either ``samples_to_slice`` or ``properties_to_slice``, or both,
    should be specified as input.

    Example: take an input :py:class:`TensorBlock` of shape (100, 1, 6), where
    there are 100 'samples', 1 'components', and 6 'properties'. Say we want to
    slice this tensor along the samples and properties dimensions. As in the
    code-block below, we can specify, for example, 4 samples and 3 properties
    indices to keep. The returned :py:class:`TensorBlock` will have shape (4, 1,
    3).

    .. code-block:: python

        sliced_block = slice_block(
            block,
            samples=Labels(
                names=["structure", "center"],
                values=np.array(
                    [[0, 1], [0, 6], [1, 6], [3, 16]]
                ),  # must be a 2D-array
            )
            properties=Labels(
                names=["n",],  # radial channel
                values=np.array([[3], [4], [5]]),
            )
        )

    For the empty blocks that may be returned, although there will be no actual
    values in its TensorBlock.values tensor, the shape of this tensor will be
    non-zero in the dimensions that haven't been sliced. This is created by
    slicing the input TensorBlock, as opposed to just returning an
    artificially-created empty one (with no shape or dimensions), and is
    intentional. It allows the slicing of dimensions to be tracked.

    For instance, if a TensorBlock of shape (52, 1, 5) is passed, and only some
    samples are specified to be sliced but none of these appear in the input
    TensorBlock, the returned TensorBlock values array will be empty, but its
    shape will be (0, 1, 5) - i.e. the samples dimension has been sliced to zero
    but the components and properties dimensions remain in-tact. The same logic
    applies to any Gradient TensorBlocks the input TensorBlock may have
    associated with it.

    :param block: the input :py:class:`TensorBlock` to be sliced.
    :param samples_to_slice: a :py:class:`Labels` object containing the names
        and indices of samples to keep in the sliced output
        :py:class:`TensorBlock`. Default value of None indicates no slicing
        along the samples dimension should occur.
    :param properties_to_slice: a :py:class:`Labels` object containing the names
        and indices of properties to keep in the sliced output
        :py:class:`TensorBlock`. Default value of None indicates no slicing
        along the properties dimension should occur.

    :return new_block: a :py:class:`TensorBlock` that corresponds to the sliced
        input.
    """

    # Perform input checks
    if not isinstance(block, TensorBlock):
        raise TypeError(
            "the input tensor must be a `TensorBlock` object, if you want to "
            "to slice a `TensorMap`, use `slice()` instead"
        )

    _check_slice_types(
        samples_to_slice=samples_to_slice,
        properties_to_slice=properties_to_slice,
        sample_names=block.samples.names,
        property_names=block.properties.names,
    )

    # Slice TensorMap and issue warning if the output block is empty
    sliced_block = _slice_block(
        block,
        samples_to_slice=samples_to_slice,
        properties_to_slice=properties_to_slice,
    )
    if np.any(np.array(sliced_block.values.shape) == 0):
        warnings.warn(
            "Your input TensorBlock is now empty, based on your choice of samples "
            + "and/or properties to slice by. "
        )

    return sliced_block


def _check_slice_types(
    samples_to_slice,
    properties_to_slice,
    sample_names,
    property_names,
):
    """Perform checks for samples_to_slice/properties_to_slice"""

    if samples_to_slice is None and properties_to_slice is None:
        raise ValueError(
            "you must specify either samples or properties (or both) to slice by"
        )

    if samples_to_slice is not None:
        if not isinstance(samples_to_slice, Labels):
            raise TypeError("samples_to_slice must be a `Labels` object")
        for name in samples_to_slice.names:
            if name not in sample_names:
                raise ValueError(
                    f"invalid sample name '{name}' which is not part of the input"
                )

    if properties_to_slice is not None:
        if not isinstance(properties_to_slice, Labels):
            raise TypeError("properties_to_slice must be a `Labels` object")

        for name in properties_to_slice.names:
            if name not in property_names:
                raise ValueError(
                    f"invalid property name '{name}' which is not part of the input"
                )


def _slice_block(
    block: TensorBlock, samples_to_slice=None, properties_to_slice=None
) -> TensorBlock:
    """
    Slices an input :py:class:`TensorBlock` along the samples and/or properties
    dimension(s). ``samples_to_slice`` and ``properties_to_slice`` are
    :py:class:`Labels` objects that specify the samples/properties
    (respectively) names and indices that should be sliced, i.e. kept in the
    output :py:class:`TensorBlock`.

    Note that either ``samples_to_slice`` or ``properties_to_slice``, or both,
    should be specified as input.

    :param block: the input :py:class:`TensorBlock` to be sliced.
    :param samples_to_slice: a :py:class:`Labels` object containing the names
        and indices of samples to keep in the sliced output
        :py:class:`TensorBlock`. Default value of None indicates no slicing
        along the samples dimension should occur.
    :param properties_to_slice: a :py:class:`Labels` object containing the names
        and indices of properties to keep in the sliced output
        :py:class:`TensorBlock`. Default value of None indicates no slicing
        along the properties dimension should occur.

    :return new_block: a :py:class:`TensorBlock` that corresponds to the sliced
        input.
    """

    new_values = block.values
    new_samples = block.samples
    new_properties = block.properties

    # Generate arrays of bools indicating which samples indices to keep upon slicing.
    if samples_to_slice is not None:
        samples = block.samples[list(samples_to_slice.names)].tolist()
        set_samples_to_slice = set(samples_to_slice.tolist())
        samples_filter = np.array(
            [sample in set_samples_to_slice for sample in samples]
        )
        new_values = new_values[samples_filter]
        new_samples = new_samples[samples_filter]

    # Generate array of bools indicating which properties indices to keep upon slicing.
    if properties_to_slice is not None:
        properties = block.properties[list(properties_to_slice.names)].tolist()
        set_properties_to_slice = set(properties_to_slice.tolist())
        properties_filter = np.array(
            [prop in set_properties_to_slice for prop in properties]
        )
        new_values = new_values[..., properties_filter]
        new_properties = new_properties[properties_filter]

    # Create a new TensorBlock, sliced along the samples and properties dimension.
    new_block = TensorBlock(
        values=new_values,
        samples=new_samples,
        components=block.components,
        properties=new_properties,
    )

    # Create a map from the previous samples indexes to the new sample indexes
    # to update the gradient samples
    if samples_to_slice is not None:
        # sample_map contains at position old_sample the index of the
        # corresponding new sample
        sample_map = np.full(shape=len(samples_filter), fill_value=-1)
        last = 0
        for i, picked in enumerate(samples_filter):
            if picked:
                sample_map[i] = last
                last += 1

    # Slice each Gradient TensorBlock and add to the new_block.
    for parameter, gradient in block.gradients():

        new_grad_data = gradient.data
        new_grad_samples = gradient.samples

        # Create a samples filter for the Gradient TensorBlock
        if samples_to_slice is not None:
            grad_samples_filter = samples_filter[gradient.samples["sample"]]
            new_grad_samples = new_grad_samples[grad_samples_filter]

            if new_grad_samples.shape[0] != 0:
                # update the "sample" column of the gradient samples
                # to refer to the new samples
                new_grad_samples = (
                    new_grad_samples.view(dtype=np.int32)
                    .reshape(new_grad_samples.shape[0], -1)
                    .copy()
                )
                new_grad_samples[:, 0] = sample_map[new_grad_samples[:, 0]]

                new_grad_samples = Labels(
                    names=gradient.samples.names,
                    values=new_grad_samples,
                )

            new_grad_data = new_grad_data[grad_samples_filter]
        if properties_to_slice is not None:
            new_grad_data = new_grad_data[..., properties_filter]

        # Add sliced Gradient to the TensorBlock
        new_block.add_gradient(
            parameter=parameter,
            samples=new_grad_samples,
            components=gradient.components,
            data=new_grad_data,
        )

    return new_block


def _slice_tensormap(
    tensormap: TensorMap, samples_to_slice=None, properties_to_slice=None
) -> TensorMap:
    """
    Slices an input :py:class:`TensorMap` along the samples and/or properties
    dimension(s). ``samples_to_slice`` and ``properties_to_slice`` are
    :py:class:`Labels` objects that specify the samples/properties
    (respectively) names and indices that should be sliced, i.e. kept in the
    output tensor.

    Note that either ``samples_to_slice`` or ``properties_to_slice``, or both,
    should be specified as input.

    :param tensor: the input :py:class:`TensorMap` to be sliced.
    :param samples_to_slice: a :py:class:`Labels` object containing the names
        and indices of samples to keep in the sliced :py:class:`TensorBlock`
        output, or each of the sliced :py:class:`TensorBlock` objects of the
        output :py:class:`TensorMap`. Default value of None indicates no slicing
        along the samples dimension should occur.
    :param properties_to_slice: a :py:class:`Labels` object containing the names
        and indices of properties to keep in the sliced :py:class:`TensorBlock`
        output, or each of the sliced :py:class:`TensorBlock` objects of the
        output :py:class:`TensorMap`. Default value of None indicates no slicing
        along the properties dimension should occur.

    :return: a :py:class:`TensorMap` that corresponds to the sliced input
        tensor.
    """

    # Iterate over, and slice, each block (+ gradients) of the input TensorMap in turn.
    return TensorMap(
        keys=tensormap.keys,
        blocks=[
            _slice_block(block, samples_to_slice, properties_to_slice)
            for _, block in tensormap
        ],
    )
