from typing import List, Union

from . import _dispatch
from ._backend import (
    Labels,
    LabelsValues,
    TensorBlock,
    TensorMap,
    is_metatensor_class,
    torch_jit_is_scripting,
    torch_jit_script,
)
from ._dispatch import TorchTensor


SliceSelection = Union[List[int], LabelsValues, Labels]


def _slice_block(
    block: TensorBlock, axis: str, selection: SliceSelection
) -> TensorBlock:
    if isinstance(selection, list):
        selected = _dispatch.int_array_like(selection, block.samples.values)
    elif isinstance(selection, LabelsValues):
        selected = selection
    else:
        if not torch_jit_is_scripting():
            # This should already have been checked
            assert is_metatensor_class(selection, Labels)

        if axis == "samples":
            selected = block.samples.select(selection)
        else:
            assert axis == "properties"
            selected = block.properties.select(selection)

    if axis == "samples":
        bool_array = _dispatch.bool_array_like([], block.properties.values)
        mask = _dispatch.zeros_like(bool_array, [len(block.samples)])
        mask[selected] = True

        new_block = TensorBlock(
            values=block.values[selected],
            samples=Labels(block.samples.names, block.samples.values[selected]),
            components=block.components,
            properties=block.properties,
        )

        # Create a map from the previous samples indexes to the new sample indexes
        # to update the gradient samples

        # sample_map contains at position `old_sample` the index of the
        # corresponding `new_sample`
        sample_map = _dispatch.int_array_like(
            int_list=[-1] * len(block.samples),
            like=block.samples.values,
        )

        sample_map[selected] = _dispatch.int_array_like(
            list(range(len(selected))),
            like=block.samples.values,
        )

        for parameter, gradient in block.gradients():
            if len(gradient.gradients_list()) != 0:
                raise NotImplementedError("gradients of gradients are not supported")

            sample_column = gradient.samples.column("sample")
            if not isinstance(gradient.samples.values, TorchTensor) and isinstance(
                mask, TorchTensor
            ):
                # Torch complains if `sample_column` is numpy since it tries to convert
                # it to a Tensor, but the numpy array is read-only. Making a copy
                # removes the read-only marker
                sample_column = sample_column.copy()

            # Create a samples filter for the Gradient TensorBlock
            grad_samples_mask = mask[_dispatch.to_index_array(sample_column)]

            new_grad_samples_values = _dispatch.mask(
                gradient.samples.values, 0, grad_samples_mask
            )

            if new_grad_samples_values.shape[0] != 0:
                # update the "sample" column of the gradient samples
                # to refer to the new samples
                new_grad_samples_values[:, 0] = sample_map[
                    _dispatch.to_index_array(new_grad_samples_values[:, 0])
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
        bool_array = _dispatch.bool_array_like([], block.properties.values)
        mask = _dispatch.zeros_like(bool_array, [len(block.properties)])
        mask[selected] = True

        new_values = _dispatch.mask(block.values, len(block.values.shape) - 1, mask)
        new_properties = Labels(block.properties.names, block.properties.values[mask])

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
                gradient.values, len(gradient.values.shape) - 1, mask
            )

            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=new_grad_values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=new_properties,
                ),
            )

    return new_block


def _check_slice_args(
    block: TensorBlock,
    axis: str,
    selection: SliceSelection,
):
    """Checks the arguments passed to :py:func:`slice` and :py:func:`slice_block`."""
    if axis not in ["samples", "properties"]:
        raise ValueError(
            f"``axis``: {axis} is not known as a slicing axis. Please use"
            "'samples' or 'properties'"
        )

    if isinstance(selection, LabelsValues):
        if len(selection.shape) > 1:
            raise ValueError("`selection` must be a 1-D array of integers")
    elif not isinstance(selection, list):
        if not torch_jit_is_scripting():
            if not is_metatensor_class(selection, Labels):
                raise TypeError(
                    "`selection` must be metatensor Labels, an array "
                    + f"or List[int], not {type(selection)}"
                )

        if axis == "samples":
            s_names = block.samples.names
            for name in selection.names:
                if name not in s_names:
                    raise ValueError(
                        f"invalid sample name '{name}' which is not part of the input"
                    )
        else:
            assert axis == "properties"
            p_names = block.properties.names
            for name in selection.names:
                if name not in p_names:
                    raise ValueError(
                        f"invalid property name '{name}' which is not part of the input"
                    )


@torch_jit_script
def slice(tensor: TensorMap, axis: str, selection: Labels) -> TensorMap:
    """
    Slice a :py:class:`TensorMap` along either the ``"samples"`` or ``"properties"`
    axis. The ``selection`` specifies which samples/properties (respectively) should be
    kept in the output :py:class:`TensorMap`.

    This function will return a :py:class:`TensorMap` whose blocks are of equal or
    smaller dimensions (due to slicing) than those of the input. However, the returned
    :py:class:`TensorMap` will be returned with the same number of blocks and the
    corresponding keys as the input. If any block upon slicing is reduced to nothing,
    i.e. in the case that it has none of the specified ``selection`` along the
    ``"samples"`` or ``"properties"`` ``axis``, an empty block (i.e. a block with one of
    the dimension set to 0) will used for this key, and a warning will be emitted.

    See the documentation for the :py:func:`slice_block` function to see how an
    individual :py:class:`TensorBlock` is sliced.

    :param tensor: the input :py:class:`TensorMap` to be sliced.
    :param axis: a :py:class:`str` indicating the axis along which slicing should occur.
        Should be either "samples" or "properties".
    :param selection: a :py:class:`Labels` object containing a selection for the
        ``"samples"`` or ``"properties"`` to keep in the sliced :py:class:`TensorMap`,
        or an array or ``List[int]`` indicating the raw indices that should be kept.
        When using :py:class:`Labels` selection, only a subset of the corresponding
        dimension names can be specified, and any entry with matching values will be
        selected.

    :return: a :py:class:`TensorMap` that corresponds to the sliced input tensor.
    """
    # Check input args
    if not torch_jit_is_scripting():
        if not is_metatensor_class(tensor, TensorMap):
            raise TypeError(
                f"`tensor` must be a metatensor TensorMap, not {type(tensor)}"
            )

    _check_slice_args(tensor.block(0), axis=axis, selection=selection)

    return TensorMap(
        keys=tensor.keys,
        blocks=[
            _slice_block(tensor[tensor.keys.entry(i)], axis, selection)
            for i in range(len(tensor.keys))
        ],
    )


@torch_jit_script
def slice_block(
    block: TensorBlock, axis: str, selection: SliceSelection
) -> TensorBlock:
    """
    Slices a :py:class:`TensorBlock` along either the ``"samples"`` or ``"properties"``
    axis. The ``selection`` specifies which samples/properties (respectively) should be
    kept in the output :py:class:`TensorMap`.

    If none of the entries in ``selection`` can be found in the ``block``, the dimension
    corresponding to ``axis`` will be sliced to 0, and the returned block with have a
    shape of either ``(0, n_components, n_properties)`` or ``(n_samples, n_components,
    0)``.

    :param block: the input :py:class:`TensorBlock` to be sliced.
    :param axis: a :py:class:`str` indicating the axis along which slicing should occur.
        Should be either "samples" or "properties".
    :param selection: a :py:class:`Labels` object containing a selection for the
        ``"samples"`` or ``"properties"`` to keep in the sliced :py:class:`TensorBlock`,
        or an array or ``List[int]`` indicating the raw indices that should be kept.
        When using :py:class:`Labels` selection, only a subset of the corresponding
        dimension names can be specified, and any entry with matching values will be
        selected.

    :return new_block: a :py:class:`TensorBlock` that corresponds to the sliced input.
    """
    if not torch_jit_is_scripting():
        if not is_metatensor_class(block, TensorBlock):
            raise TypeError(
                f"`block` must be a metatensor TensorBlock, not {type(block)}"
            )

    _check_slice_args(block, axis=axis, selection=selection)

    return _slice_block(
        block,
        axis=axis,
        selection=selection,
    )
