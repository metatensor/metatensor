from typing import Optional, Union

import numpy as np

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap


def slice(
    tensor: TensorMap,
    samples: Optional[Labels] = None,
    properties: Optional[Labels] = None,
) -> TensorMap:
    """Slice a :py:class:`TensorMap` along the samples and/or properties dimension(s).

    ``samples`` and ``properties`` are :py:class:`Labels` objects that specify the
    samples/properties (respectively) names and indices that should be sliced, i.e.
    kept in the output tensor.

    Note that either ``samples`` or ``properties``, or both,
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
    specified ``samples`` or ``properties``, an empty block
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
    :param samples: a :py:class:`Labels` object containing the names
        and indices of samples to keep in the each of the sliced
        :py:class:`TensorBlock` of the output :py:class:`TensorMap`. Default
        value of None indicates no slicing along the samples dimension should
        occur.
    :param properties: a :py:class:`Labels` object containing the names
        and indices of properties to keep in each of the sliced
        :py:class:`TensorBlock` of the output :py:class:`TensorMap`. Default
        value of None indicates no slicing along the properties dimension should
        occur.

    :return: a :py:class:`TensorMap` that corresponds to the sliced input
        tensor.
    """
    # Check input args
    if not isinstance(tensor, TensorMap):
        raise TypeError("``tensor`` should be an equistore ``TensorMap``")
    _check_args(tensor, samples=samples, properties=properties)

    return TensorMap(
        keys=tensor.keys,
        blocks=[_slice_block(tensor[key], samples, properties) for key in tensor.keys],
    )


def slice_block(
    block: TensorBlock,
    samples: Optional[Labels] = None,
    properties: Optional[Labels] = None,
) -> TensorBlock:
    """
    Slices an input :py:class:`TensorBlock` along the samples and/or properties
    dimension(s). ``samples`` and ``properties`` are
    :py:class:`Labels` objects that specify the samples/properties
    (respectively) names and indices that should be sliced, i.e. kept in the
    output :py:class:`TensorBlock`.

    Note that either ``samples`` or ``properties``, or both,
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
    :param samples: a :py:class:`Labels` object containing the names
        and indices of samples to keep in the sliced output
        :py:class:`TensorBlock`. Default value of None indicates no slicing
        along the samples dimension should occur.
    :param properties: a :py:class:`Labels` object containing the names
        and indices of properties to keep in the sliced output
        :py:class:`TensorBlock`. Default value of None indicates no slicing
        along the properties dimension should occur.

    :return new_block: a :py:class:`TensorBlock` that corresponds to the sliced
        input.
    """
    # Check input args
    if not isinstance(block, TensorBlock):
        raise TypeError("``block`` should be an equistore ``TensorBlock``")
    _check_args(block, samples=samples, properties=properties)

    return _slice_block(
        block,
        samples=samples,
        properties=properties,
    )


def _slice_block(
    block: TensorBlock,
    samples: Optional[Labels] = None,
    properties: Optional[Labels] = None,
) -> TensorBlock:
    """
    Slices an input :py:class:`TensorBlock` along the samples and/or properties
    dimension(s). ``samples`` and ``properties`` are
    :py:class:`Labels` objects that specify the samples/properties
    (respectively) names and indices that should be sliced, i.e. kept in the
    output :py:class:`TensorBlock`.

    Note that either ``samples`` or ``properties``, or both,
    should be specified as input.

    :param block: the input :py:class:`TensorBlock` to be sliced.
    :param samples: a :py:class:`Labels` object containing the names
        and indices of samples to keep in the sliced output
        :py:class:`TensorBlock`. Default value of None indicates no slicing
        along the samples dimension should occur.
    :param properties: a :py:class:`Labels` object containing the names
        and indices of properties to keep in the sliced output
        :py:class:`TensorBlock`. Default value of None indicates no slicing
        along the properties dimension should occur.

    :return new_block: a :py:class:`TensorBlock` that corresponds to the sliced
        input.
    """
    # Store current values for later modification
    new_values = block.values
    new_samples = block.samples
    new_properties = block.properties

    # Generate arrays of bools indicating which samples indices to keep upon slicing.
    if samples is not None:
        all_samples = block.samples[list(samples.names)].tolist()
        set_samples_to_slice = set(samples.tolist())
        samples_filter = np.array(
            [sample in set_samples_to_slice for sample in all_samples]
        )
        new_values = new_values[samples_filter]
        new_samples = new_samples[samples_filter]

    # Generate array of bools indicating which properties indices to keep upon slicing.
    if properties is not None:
        all_properties = block.properties[list(properties.names)].tolist()
        set_properties_to_slice = set(properties.tolist())
        properties_filter = np.array(
            [prop in set_properties_to_slice for prop in all_properties]
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
    if samples is not None:
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
        if samples is not None:
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
        if properties is not None:
            new_grad_data = new_grad_data[..., properties_filter]

        # Add sliced Gradient to the TensorBlock
        new_block.add_gradient(
            parameter=parameter,
            samples=new_grad_samples,
            components=gradient.components,
            data=new_grad_data,
        )

    return new_block


def _check_args(
    tensor: Union[TensorMap, TensorMap],
    samples: Optional[Labels] = None,
    properties: Optional[Labels] = None,
):
    """
    Checks the arguments passed to :py:func:`slice` and :py:func:`slice_block`.
    """
    # Get a single block
    block = tensor.block(0) if isinstance(tensor, TensorMap) else tensor
    # Check samples Labels if passed
    if samples is not None:
        # Check type
        if not isinstance(samples, Labels):
            raise TypeError("samples must be a `Labels` object")
        # Check names
        s_names = block.samples.names
        for name in samples.names:
            if name not in s_names:
                raise ValueError(
                    f"invalid sample name '{name}' which is not part of the input"
                )
    # Check properties Labels if passed
    if properties is not None:
        # Check type
        if not isinstance(properties, Labels):
            raise TypeError("properties must be a `Labels` object")
        # Check names
        p_names = block.properties.names
        for name in properties.names:
            if name not in p_names:
                raise ValueError(
                    f"invalid property name '{name}' which is not part of the input"
                )
