from . import _dispatch
from ._classes import (
    Labels,
    TensorBlock,
    TensorMap,
    check_isinstance,
    torch_jit_is_scripting,
)
from ._dispatch import TorchTensor


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
    if not torch_jit_is_scripting():
        if not check_isinstance(tensor, TensorMap):
            raise TypeError(
                f"`tensor` must be a metatensor TensorMap, not {type(tensor)}"
            )

    _check_args(tensor.block(0), axis=axis, labels=labels)

    return TensorMap(
        keys=tensor.keys,
        blocks=[
            _slice_block(tensor[tensor.keys.entry(i)], axis, labels)
            for i in range(len(tensor.keys))
        ],
    )


def slice_block(block: TensorBlock, axis: str, labels: Labels) -> TensorBlock:
    """
    Slices an input :py:class:`TensorBlock` along either the "samples" or
    "properties" `axis`.  `labels` is a :py:class:`Labels` objects that specify
    the sample/property names and indices that should be sliced, i.e. kept in
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
    if not torch_jit_is_scripting():
        if not check_isinstance(block, TensorBlock):
            raise TypeError(
                f"`block` must be a metatensor TensorBlock, not {type(block)}"
            )

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
    the sample/property names and indices that should be sliced, i.e. kept in
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

    if axis == "samples":
        # only keep the same names as `labels`
        all_samples = block.samples.view(labels.names)
        # create an arrays of bools indicating which samples indices to keep
        samples_mask = _dispatch.bool_array_like(
            [all_samples.entry(i) in labels for i in range(len(all_samples))],
            block.values,
        )
        new_values = _dispatch.mask(block.values, 0, samples_mask)
        new_samples = Labels(
            block.samples.names,
            _dispatch.mask(block.samples.values, 0, samples_mask),
        )

        new_block = TensorBlock(
            values=new_values,
            samples=new_samples,
            components=block.components,
            properties=block.properties,
        )

        # Create a map from the previous samples indexes to the new sample indexes
        # to update the gradient samples

        # sample_map contains at position old_sample the index of the
        # corresponding new sample
        sample_map = _dispatch.int_array_like(
            int_list=[-1] * len(samples_mask),
            like=samples_mask,
        )
        last = 0
        for i, picked in enumerate(samples_mask):
            if picked:
                sample_map[i] = last
                last += 1

        for parameter, gradient in block.gradients():
            if len(gradient.gradients_list()) != 0:
                raise NotImplementedError("gradients of gradients are not supported")

            sample_column = gradient.samples.column("sample")
            if not isinstance(gradient.samples.values, TorchTensor) and isinstance(
                samples_mask, TorchTensor
            ):
                # Torch complains if `sample_column` is numpy since it tries to convert
                # it to a Tensor, but the numpy array is read-only. Making a copy
                # removes the read-only marker
                sample_column = sample_column.copy()

            # Create a samples filter for the Gradient TensorBlock
            grad_samples_mask = samples_mask[sample_column]

            new_grad_samples_values = _dispatch.mask(
                gradient.samples.values, 0, grad_samples_mask
            )

            if new_grad_samples_values.shape[0] != 0:
                # update the "sample" column of the gradient samples
                # to refer to the new samples
                new_grad_samples_values[:, 0] = sample_map[
                    new_grad_samples_values[:, 0]
                ]

                new_grad_samples = Labels(
                    names=gradient.samples.names,
                    values=new_grad_samples_values,
                )
            else:
                new_grad_samples = Labels(
                    names=gradient.samples.names,
                    values=_dispatch.empty_like(
                        gradient.samples.values, [0, gradient.samples.values.shape[1]]
                    ),
                )

            new_grad_values = _dispatch.mask(gradient.values, 0, grad_samples_mask)
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

    else:
        assert axis == "properties"
        # only keep the same names as `labels`
        all_properties = block.properties.view(list(labels.names))
        # create an arrays of bools indicating which samples indices to keep
        properties_mask = _dispatch.bool_array_like(
            [all_properties.entry(i) in labels for i in range(len(all_properties))],
            block.values,
        )
        new_values = _dispatch.mask(
            block.values, len(block.values.shape) - 1, properties_mask
        )
        new_properties = Labels(
            block.properties.names,
            _dispatch.mask(block.properties.values, 0, properties_mask),
        )

        new_block = TensorBlock(
            values=new_values,
            samples=block.samples,
            components=block.components,
            properties=new_properties,
        )

        # Slice each Gradient TensorBlock and add to the new_block.
        for parameter, gradient in block.gradients():
            if len(gradient.gradients_list()) != 0:
                raise NotImplementedError("gradients of gradients are not supported")

            assert axis == "properties"
            new_grad_values = _dispatch.mask(
                gradient.values, len(gradient.values.shape) - 1, properties_mask
            )
            new_grad_samples = gradient.samples

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
    block: TensorBlock,
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

    if not torch_jit_is_scripting():
        if not check_isinstance(labels, Labels):
            raise TypeError(f"`labels` must be metatensor Labels, not {type(labels)}")

    if axis == "samples":
        s_names = block.samples.names
        for name in labels.names:
            if name not in s_names:
                raise ValueError(
                    f"invalid sample name '{name}' which is not part of the input"
                )
    else:
        assert axis == "properties"
        p_names = block.properties.names
        for name in labels.names:
            if name not in p_names:
                raise ValueError(
                    f"invalid property name '{name}' which is not part of the input"
                )
