from typing import Union

import numpy as np

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap


def slice(tensor: TensorMap, axis: str, labels: Labels) -> TensorMap:
    """
    Slice a :py:class:`TensorMap` along either the "samples" or "properties" `axis`.

    `labels` is a :py:class:`Labels` objects that specifies the
    samples/properties (respectively) names and indices that should be sliced,
    i.e. kept in the output tensor. For axis, either "samples" or "properties"
    should be specified.

    .. code-block:: python

        samples = Labels(
            names=["structure", "center"],
            values=np.array([[0, 1], [0, 6], [1, 6], [3, 16]]),  # must be a 2D-array
        )
        properties = Labels(
            names=["n"],  # radial channel
            values=np.array([[3], [4], [5]]),
        )
        sliced_tensor_samples = slice(
            tensor,
            axis="samples",
            labels=samples,
        )
        sliced_tensor_properties = slice(
            tensor,
            axis="properties",
            labels=properties,
        )

    Also note that this function will return a :py:class:`TensorMap` whose
    blocks are of equal or smaller dimensions (due to slicing) than those of the
    input. However, the returned :py:class:`TensorMap` will be returned with the
    same number of blocks and the corresponding keys as the input. If any block
    upon slicing is reduced to nothing, i.e. in the case that it has none of the
    specified `labels` along the "samples" or "properties" `axis`, an empty
    block will be returned but will still be accessible by its key. User
    warnings will be issued if any blocks are sliced to contain no values.

    For the empty blocks that may be returned, although there will be no actual
    values in its ``TensorBlock.values`` array, the shape of this array will be
    non-zero in the dimensions that haven't been sliced. This allows the slicing
    of dimensions to be tracked.

    For example, if a TensorBlock of shape (52, 1, 5) is passed, and only some
    samples are specified to be sliced but none of these appear in the input
    :py:class:`TensorBlock`, the returned block values array will be empty, but
    its shape will be (0, 1, 5) - i.e. the samples dimension has been sliced to
    zero but the components and properties dimensions remain in-tact. The same
    logic applies to any Gradient TensorBlocks the input TensorBlock may have
    associated with it.

    See the documentation for the :py:func:`slice_block` function to see how an
    individual :py:class:`TensorBlock` is sliced.

    :param tensor: the input :py:class:`TensorMap` to be sliced.
    :param axis: a :py:class:`str` indicating the axis along which slicing
        should occur. Should be either "samples" or "properties".
    :param labels: a :py:class:`Labels` object containing the names and indices
        of the "samples" or "properties" to keep in each of the sliced
        :py:class:`TensorBlock` of the output :py:class:`TensorMap`.

    :return: a :py:class:`TensorMap` that corresponds to the sliced input
        tensor.
    """
    # Check input args
    if not isinstance(tensor, TensorMap):
        raise TypeError("``tensor`` should be an equistore ``TensorMap``")
    _check_args(tensor, axis=axis, labels=labels)

    return TensorMap(
        keys=tensor.keys,
        blocks=[_slice_block(tensor[key], axis, labels) for key in tensor.keys],
    )


def slice_block(block: TensorBlock, axis: str, labels: Labels) -> TensorBlock:
    """
    Slices an input :py:class:`TensorBlock` along either the "samples" or
    "properties" `axis`.  `labels` is a :py:class:`Labels` objects that specify
    the samples/properties names and indices that should be sliced, i.e. kept in
    the output :py:class:`TensorBlock`.

    Example: take an input :py:class:`TensorBlock` of shape (100, 1, 6), where
    there are 100 'samples', 1 'components', and 6 'properties'. Say we want to
    slice this tensor along the samples and properties dimensions. As in the
    code-block below, we can specify, for example, 4 samples and 3 properties
    indices to keep. The returned :py:class:`TensorBlock` will have shape (4, 1,
    3).

    .. code-block:: python

        samples = Labels(
            names=["structure", "center"],
            values=np.array([[0, 1], [0, 6], [1, 6], [3, 16]]),  # must be a 2D-array
        )
        properties = Labels(
            names=["n"],  # radial channel
            values=np.array([[3], [4], [5]]),
        )
        sliced_block_samples = slice_block(
            block,
            axis="samples",
            labels=samples,
        )
        sliced_block_properties = slice_block(
            block,
            axis="properties",
            labels=properties,
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
    :param axis: a :py:class:`str` indicating the axis along which slicing
        should occur. Should be either "samples" or "properties".
    :param labels: a :py:class:`Labels` object containing the names and indices
        of the "samples" or "properties" to keep in the sliced output
        :py:class:`TensorBlock`.

    :return new_block: a :py:class:`TensorBlock` that corresponds to the sliced
        input.
    """
    # Check input args
    if not isinstance(block, TensorBlock):
        raise TypeError("``block`` should be an equistore ``TensorBlock``")
    _check_args(block, axis=axis, labels=labels)

    return _slice_block(
        block,
        axis=axis,
        labels=labels,
    )


def _slice_block(block: TensorBlock, axis: str, labels: Labels) -> TensorBlock:
    """
    Slices an input :py:class:`TensorBlock` along either the "samples" or
    "properties" `axis`. `labels` is :py:class:`Labels` object that specifies
    the samples/properties names and indices that should be sliced, i.e. kept in
    the output :py:class:`TensorBlock`.

    :param block: the input :py:class:`TensorBlock` to be sliced.
    :param axis: a :py:class:`str` object containing `samples` or `properties`
        indicating the direction of slicing.
    :param labels: a :py:class:`Labels` object containing the names and indices
        of samples/properties to keep in the sliced output
        :py:class:`TensorBlock`.

    :return new_block: a :py:class:`TensorBlock` that corresponds to the sliced
        input.
    """
    # Store current values for later modification
    new_values = block.values
    if axis == "samples":
        new_samples = block.samples
    else:
        assert axis == "properties"
        new_properties = block.properties

    # Generate arrays of bools indicating which samples indices to keep upon slicing.
    if axis == "samples":
        all_samples = block.samples[list(labels.names)].tolist()
        set_samples_to_slice = set(labels.tolist())
        samples_filter = np.array(
            [sample in set_samples_to_slice for sample in all_samples]
        )
        new_values = new_values[samples_filter]
        new_samples = new_samples[samples_filter]

    # Generate array of bools indicating which properties indices to keep upon slicing.
    else:
        assert axis == "properties"
        all_properties = block.properties[list(labels.names)].tolist()
        set_properties_to_slice = set(labels.tolist())
        properties_filter = np.array(
            [prop in set_properties_to_slice for prop in all_properties]
        )
        new_values = new_values[..., properties_filter]
        new_properties = new_properties[properties_filter]

    # Create a new TensorBlock, sliced along the samples and properties dimension.
    if axis == "samples":
        new_block = TensorBlock(
            values=new_values,
            samples=new_samples,
            components=block.components,
            properties=block.properties,
        )
    else:
        assert axis == "properties"
        new_block = TensorBlock(
            values=new_values,
            samples=block.samples,
            components=block.components,
            properties=new_properties,
        )

    # Create a map from the previous samples indexes to the new sample indexes
    # to update the gradient samples
    if axis == "samples":
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
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        new_grad_values = gradient.values
        new_grad_samples = gradient.samples

        # Create a samples filter for the Gradient TensorBlock
        if axis == "samples":
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

            new_grad_values = new_grad_values[grad_samples_filter]
        else:
            assert axis == "properties"
            new_grad_values = new_grad_values[..., properties_filter]

        # Add sliced gradient to the TensorBlock
        new_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=new_grad_values,
                samples=new_grad_samples,
                components=gradient.components,
                properties=new_block.properties,
            ),
        )

    return new_block


def _check_args(
    tensor: Union[TensorMap, TensorMap],
    axis: str,
    labels: Labels,
):
    """
    Checks the arguments passed to :py:func:`slice` and :py:func:`slice_block`.
    """
    # check axis
    if axis not in ["samples", "properties"]:
        raise ValueError(
            f"``axis``: {axis} is not known as a slicing axis. Please use"
            "'samples' or 'properties'"
        )
    # Get a single block
    block = tensor.block(0) if isinstance(tensor, TensorMap) else tensor
    # Check if labels are Labels if passed
    # Check type
    if not isinstance(labels, Labels):
        print(type(labels))
        raise TypeError("labels must be a `Labels` object")
    if axis == "samples":
        # Check names
        s_names = block.samples.names
        for name in labels.names:
            if name not in s_names:
                raise ValueError(
                    f"invalid sample name '{name}' which is not part of the input"
                )
    # Check properties Labels if passed
    else:
        assert axis == "properties"
        # Check names
        p_names = block.properties.names
        for name in labels.names:
            if name not in p_names:
                raise ValueError(
                    f"invalid property name '{name}' which is not part of the input"
                )
