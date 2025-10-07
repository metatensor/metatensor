from typing import List, Optional, Union

from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    isinstance_metatensor,
    torch_jit_is_scripting,
    torch_jit_script,
)
from ._utils import (
    _check_blocks_raise,
    _check_same_gradients_raise,
    _check_same_keys_raise,
)


def _unique_str(str_list: List[str]):
    unique_strings: List[str] = []
    for string in str_list:
        if string not in unique_strings:
            unique_strings.append(string)
    return unique_strings


def _tensors_intersection(tensors: List[TensorMap]) -> List[TensorMap]:
    """Create a new tensors list where keys are based on the intersection from all
    tensors.

    Blocks corresponding to keys that are not present in all tensor will be discarded.
    """
    # Construct a Labels object with intersected keys
    all_keys = tensors[0].keys
    for tensor in tensors[1:]:
        all_keys = all_keys.intersection(tensor.keys)

    # Create new blocks and discard bocks not present in all_keys
    new_tensors: List[TensorMap] = []
    for tensor in tensors:
        new_blocks: List[TensorBlock] = []
        for i_key in range(all_keys.values.shape[0]):
            new_blocks.append(tensor.block(all_keys.entry(i_key)).copy())
        new_tensors.append(TensorMap(keys=all_keys, blocks=new_blocks))

    return new_tensors


def _tensors_union(tensors: List[TensorMap], axis: str) -> List[TensorMap]:
    """Create a new tensors list where keys are based on the union from all tensors.

    Missing keys will be filled by empty blocks having containing no labels in the
    ``axis`` dimension.
    """
    # Construct a Labels object with all keys

    all_keys = tensors[0].keys
    for tensor in tensors[1:]:
        all_keys = all_keys.union(tensor.keys)

    # Create empty blocks for missing keys for each TensorMap
    new_tensors: List[TensorMap] = []
    for tensor in tensors:
        _, map, _ = all_keys.intersection_and_mapping(tensor.keys)

        missing_keys = Labels(
            names=tensor.keys.names, values=all_keys.values[map == -1]
        )

        new_keys = tensor.keys.union(missing_keys)
        new_blocks = [block.copy() for block in tensor.blocks()]

        for i_key in range(missing_keys.values.shape[0]):
            key = missing_keys.entry(i_key)
            # Find corresponding block with the missing key
            reference_block: Union[None, TensorBlock] = None
            for reference_tensor in tensors:
                if key in reference_tensor.keys:
                    reference_block = reference_tensor.block(key)
                    break

            # There should be a block with the key otherwise we did something wrong
            assert reference_block is not None

            # Construct new block with zero samples based on the metadata of
            # reference_block
            if axis == "samples":
                values = _dispatch.empty_like(
                    array=reference_block.values,
                    shape=(0,) + reference_block.values.shape[1:],
                )
                samples = Labels(
                    names=reference_block.samples.names,
                    values=_dispatch.empty_like(
                        reference_block.samples.values,
                        (0, len(reference_block.samples.names)),
                    ),
                )
                properties = reference_block.properties
            else:
                assert axis == "properties"
                values = _dispatch.empty_like(
                    array=reference_block.values,
                    shape=reference_block.values.shape[:-1] + (0,),
                )
                samples = reference_block.samples
                properties = Labels(
                    names=reference_block.properties.names,
                    values=_dispatch.empty_like(
                        reference_block.properties.values,
                        (0, len(reference_block.properties.names)),
                    ),
                )

            new_block = TensorBlock(
                values=values,
                samples=samples,
                components=reference_block.components,
                properties=properties,
            )

            for parameter, gradient in reference_block.gradients():
                if len(gradient.gradients_list()) != 0:
                    raise NotImplementedError(
                        "gradients of gradients are not supported"
                    )

                if axis == "samples":
                    values = _dispatch.empty_like(
                        array=gradient.values,
                        shape=(0,) + gradient.values.shape[1:],
                    )
                    gradient_samples = Labels(
                        names=gradient.samples.names,
                        values=_dispatch.empty_like(
                            gradient.samples.values, (0, len(gradient.samples.names))
                        ),
                    )
                else:
                    values = _dispatch.empty_like(
                        array=gradient.values,
                        shape=gradient.values.shape[:-1] + (0,),
                    )
                    gradient_samples = gradient.samples

                new_block.add_gradient(
                    parameter=parameter,
                    gradient=TensorBlock(
                        values=values,
                        samples=gradient_samples,
                        components=gradient.components,
                        properties=properties,
                    ),
                )

            new_blocks.append(new_block)

        new_tensors.append(TensorMap(keys=new_keys, blocks=new_blocks))

    return new_tensors


def _join_block_samples(
    blocks: List[TensorBlock],
    fname: str,
    add_dimension: Optional[str],
) -> TensorBlock:
    """
    Actual implementation of block joining along samples.

    :param blocks: blocks to join
    :param fname: name of the function calling ``_join_block_samples``, to be used in
        error messages
    :param add_dimension: name of the dimension to add to the samples
    """

    assert len(blocks) != 0
    if len(blocks) == 1:
        return blocks[0]

    first_block = blocks[0]
    n_joined_samples = 0
    for block in blocks:
        n_joined_samples += block.values.shape[0]
        _check_blocks_raise(first_block, block, fname, ["components", "properties"])
        _check_same_gradients_raise(first_block, block, fname, ["components"])

    shape = list(first_block.values.shape)
    shape[0] = n_joined_samples

    new_values = _dispatch.empty_like(first_block.values, shape=shape)

    first_sample_size = first_block.samples.values.shape[1]
    if add_dimension is None:
        samples_size = first_sample_size
    else:
        samples_size = first_sample_size + 1

    new_samples = _dispatch.empty_like(
        first_block.samples.values,
        shape=[n_joined_samples, samples_size],
    )

    start = 0
    for block_i, block in enumerate(blocks):
        stop = start + len(block.samples)
        new_values[start:stop] = block.values[:]

        new_samples[start:stop, :first_sample_size] = block.samples.values[:, :]
        if add_dimension is not None:
            new_samples[start:stop, first_sample_size] = block_i

        start = stop

    new_samples_names = first_block.samples.names
    if add_dimension is not None:
        new_samples_names.append(add_dimension)

    new_block = TensorBlock(
        new_values,
        Labels(new_samples_names, new_samples),
        first_block.components,
        first_block.properties,
    )

    for parameter in first_block.gradients_list():
        first_gradient = first_block.gradient(parameter)
        gradients: List[TensorBlock] = []
        n_gradient_samples = 0
        for block in blocks:
            gradient = block.gradient(parameter)
            if len(gradient.gradients_list()) != 0:
                raise NotImplementedError(
                    "gradients of gradients are not yet supported"
                )

            n_gradient_samples += len(gradient.samples)
            gradients.append(gradient)

        shape = list(first_gradient.values.shape)
        shape[0] = n_gradient_samples

        new_values = _dispatch.empty_like(first_gradient.values, shape=shape)

        new_samples = _dispatch.empty_like(
            first_gradient.samples.values,
            shape=[n_gradient_samples, first_gradient.samples.values.shape[1]],
        )

        start = 0
        sample_shift = 0
        for block, gradient in zip(blocks, gradients, strict=False):
            stop = start + len(gradient.samples)
            new_values[start:stop] = gradient.values[:]

            new_samples[start:stop] = gradient.samples.values[:]
            # update the "sample" dimension, matching the shift in the values
            new_samples[start:stop, 0] += sample_shift
            sample_shift += len(block.samples)

            start = stop

        new_gradient = TensorBlock(
            new_values,
            Labels(first_gradient.samples.names, new_samples),
            first_gradient.components,
            first_gradient.properties,
        )

        new_block.add_gradient(parameter, new_gradient)

    return new_block


def _join_block_properties(
    blocks: List[TensorBlock],
    fname: str,
    add_dimension: Optional[str],
) -> TensorBlock:
    """
    Actual implementation of block joining along properties.

    :param blocks: blocks to join
    :param fname: name of the function calling ``_join_block_properties``, to be used in
        error messages
    :param add_dimension: name of the dimension to add to the properties
    """

    assert len(blocks) != 0
    if len(blocks) == 1:
        return blocks[0]

    first_block = blocks[0]
    property_names = first_block.properties.names
    has_different_property_names = False
    n_joined_properties = 0
    for block in blocks:
        n_joined_properties += block.values.shape[-1]
        _check_blocks_raise(first_block, block, fname, ["samples", "components"])
        _check_same_gradients_raise(first_block, block, fname, ["components"])

        if block.properties.names != property_names:
            has_different_property_names = True

    shape = list(first_block.values.shape)
    shape[-1] = n_joined_properties

    new_values = _dispatch.empty_like(first_block.values, shape=shape)

    first_properties_size = first_block.properties.values.shape[1]
    if add_dimension is None:
        properties_size = first_properties_size
    else:
        properties_size = first_properties_size + 1

    if has_different_property_names:
        if add_dimension is not None:
            raise ValueError(
                "We can not add an extra dimension to properties when the inputs "
                "have different property names"
            )

        new_properties_values = _dispatch.empty_like(
            first_block.samples.values, shape=[n_joined_properties, 2]
        )
    else:
        new_properties_values = _dispatch.empty_like(
            first_block.samples.values, shape=[n_joined_properties, properties_size]
        )

    start = 0
    for block_i, block in enumerate(blocks):
        stop = start + len(block.properties)
        new_values[..., start:stop] = block.values[..., :]

        if has_different_property_names:
            new_properties_values[start:stop, 0] = block_i
            new_properties_values[start:stop, 1] = _dispatch.int_array_like(
                list(range(len(block.properties))), block.properties.values
            )
        else:
            new_properties_values[start:stop, :first_properties_size] = (
                block.properties.values[:]
            )

            if add_dimension is not None:
                new_properties_values[start:stop, first_properties_size] = block_i

        start = stop

    # finalize the new properties
    if has_different_property_names:
        new_properties = Labels(
            names=["joined_index", "property"],
            values=new_properties_values,
        )
    else:
        new_properties_names = first_block.properties.names
        if add_dimension is not None:
            new_properties_names.append(add_dimension)
        new_properties = Labels(new_properties_names, new_properties_values)

    new_block = TensorBlock(
        new_values,
        first_block.samples,
        first_block.components,
        new_properties,
    )

    for parameter in first_block.gradients_list():
        first_gradient = first_block.gradient(parameter)
        gradients: List[TensorBlock] = []
        joined_gradient_samples = first_gradient.samples
        for block in blocks:
            gradient = block.gradient(parameter)
            if len(gradient.gradients_list()) != 0:
                raise NotImplementedError(
                    "gradients of gradients are not yet supported"
                )

            joined_gradient_samples = joined_gradient_samples.union(gradient.samples)
            gradients.append(gradient)

        shape = list(first_gradient.values.shape)
        shape[0] = len(joined_gradient_samples)
        shape[-1] = len(new_properties)

        # we need to use `zeros_like` instead of `empty_like`, because some
        # gradients might be missing (i.e. implicitly zero) in some input blocks
        new_values = _dispatch.zeros_like(first_gradient.values, shape=shape)

        start = 0
        for gradient in gradients:
            stop = start + len(gradient.properties)
            # find where we should put the current gradients in the joined samples
            # we can not get the mapping in the first loop over gradients above since
            # `joined_gradient_samples` could still change
            _, _, mapping = joined_gradient_samples.union_and_mapping(gradient.samples)
            new_values[mapping, ..., start:stop] = gradient.values
            start = stop

        new_gradient = TensorBlock(
            new_values,
            joined_gradient_samples,
            first_gradient.components,
            new_properties,
        )

        new_block.add_gradient(parameter, new_gradient)

    return new_block


@torch_jit_script
def join(
    tensors: List[TensorMap],
    axis: str,
    different_keys: str = "error",
    add_dimension: Optional[str] = None,
) -> TensorMap:
    """Join a sequence of :py:class:`TensorMap` with similar keys along an axis.

    The ``axis`` parameter specifies the type of joining: with ``axis='properties'`` the
    ``tensors`` will be joined along the ``properties`` and for ``axis='samples'`` they
    will be joined along the ``samples``.

    :param tensors: sequence of :py:class:`TensorMap` to join
    :param axis: Along which axis the :py:class:`TensorMap`s should be joined. This can
        be either ``'properties'`` or ``'samples'``.
    :param different_keys: Method to handle different keys between the tensors. For
        ``"error"`` keys in all tensors have to be the same. For ``"intersection"`` only
        blocks present in all tensors will be taken into account. For ``"union"``
        missing keys will be treated like if they where associated with an empty block.
    :param add_dimension: Add an the extra dimension to the joined labels with the given
        name. See examples for the case where this is applicable. The dimension forms
        the last dimension of the joined labels.
    :return: The joined :py:class:`TensorMap` with more properties or samples than the
        inputs.

    Examples
    --------

    The first use case for this function is when joining ``TensorMap`` with the same
    labels names (either along ``samples`` or ``properties``):

    >>> import numpy as np
    >>> import metatensor as mts
    >>> from metatensor import Labels, TensorBlock, TensorMap

    >>> values = np.array([[1.1, 2.1, 3.1]])
    >>> samples = Labels("sample", np.array([[0]]))

    Define two disjoint set of :py:class:`Labels`.

    >>> properties_1 = Labels("n", np.array([[0], [2], [3]]))
    >>> properties_2 = Labels("n", np.array([[1], [4], [5]]))

    >>> block_1 = TensorBlock(
    ...     values=values,
    ...     samples=Labels.single(),
    ...     components=[],
    ...     properties=properties_1,
    ... )
    >>> block_2 = TensorBlock(
    ...     values=values,
    ...     samples=Labels.single(),
    ...     components=[],
    ...     properties=properties_2,
    ... )

    >>> tensor_1 = TensorMap(keys=Labels.single(), blocks=[block_1])
    >>> tensor_2 = TensorMap(keys=Labels.single(), blocks=[block_2])

    joining along the properties leads to

    >>> joined_tensor = mts.join([tensor_1, tensor_2], axis="properties")
    >>> joined_tensor[0].properties
    Labels(
        n
        0
        2
        3
        1
        4
        5
    )

    Second, if the labels names are the same but the values are not unique, you can ask
    to add an extra dimension to the labels when joining with ``add_dimension``, thus
    creating unique values

    >>> properties_3 = Labels("n", np.array([[0], [2], [3]]))

    ``properties_3`` has the same name and also shares values with ``properties_1`` as
    defined above.

    >>> block_3 = TensorBlock(
    ...     values=values,
    ...     samples=Labels.single(),
    ...     components=[],
    ...     properties=properties_3,
    ... )
    >>> tensor_3 = TensorMap(keys=Labels.single(), blocks=[block_3])

    joining along properties leads to

    >>> joined_tensor = mts.join(
    ...     [tensor_1, tensor_3], axis="properties", add_dimension="tensor"
    ... )
    >>> joined_tensor[0].properties
    Labels(
        n  tensor
        0    0
        2    0
        3    0
        0    1
        2    1
        3    1
    )

    Finally, when joining along properties, if different ``TensorMap`` have different
    property names, we'll re-create new properties labels containing the original tensor
    index and the corresponding property index. This does not apply when joining along
    samples.

    >>> properties_4 = Labels(["a", "b"], np.array([[0, 0], [1, 2], [1, 3]]))

    ``properties_4`` has the different names compared to ``properties_1`` defined above.

    >>> block_4 = TensorBlock(
    ...     values=values,
    ...     samples=Labels.single(),
    ...     components=[],
    ...     properties=properties_4,
    ... )
    >>> tensor_4 = TensorMap(keys=Labels.single(), blocks=[block_4])

    joining along properties leads to

    >>> joined_tensor = mts.join([tensor_1, tensor_4], axis="properties")
    >>> joined_tensor[0].properties
    Labels(
        joined_index  property
             0           0
             0           1
             0           2
             1           0
             1           1
             1           2
    )
    """
    if not torch_jit_is_scripting():
        if not isinstance(tensors, (list, tuple)):
            raise TypeError(f"`tensors` must be a list or a tuple, not {type(tensors)}")

        for tensor in tensors:
            if not isinstance_metatensor(tensor, "TensorMap"):
                raise TypeError(
                    "`tensors` elements must be metatensor TensorMap, "
                    f"not {type(tensor)}"
                )

    if len(tensors) == 0:
        raise ValueError("provide at least one `TensorMap` for joining")

    if axis not in ("samples", "properties"):
        raise ValueError(
            "Only `'properties'` or `'samples'` are "
            "valid values for the `axis` parameter."
        )

    if len(tensors) == 1:
        return tensors[0]

    if different_keys == "error":
        for ts_to_join in tensors[1:]:
            _check_same_keys_raise(tensors[0], ts_to_join, "join")
    elif different_keys == "intersection":
        tensors = _tensors_intersection(tensors)
    elif different_keys == "union":
        tensors = _tensors_union(tensors, axis=axis)
    else:
        raise ValueError(
            f"'{different_keys}' is not a valid option for `different_keys`. Choose "
            "either 'error', 'intersection' or 'union'."
        )

    # Deduce if sample/property names are the same in all tensors.
    # If this is not the case we have to change unify the corresponding labels later.
    if axis == "samples":
        names_list = [tensor.sample_names for tensor in tensors]

        names_list_flattened: List[str] = []
        for names in names_list:
            names_list_flattened += names

        unique_names = _unique_str(names_list_flattened)
        length_equal = [len(unique_names) == len(names) for names in names_list]
        sample_names_are_same = sum(length_equal) == len(length_equal)

        if not sample_names_are_same:
            # It's fine to lose metadata for the properties, less so for the samples
            raise ValueError(
                "Different tensor have different sample names in `join`. "
                "Joining along samples with different sample names will lose "
                "information and is not supported."
            )

    keys = tensors[0].keys
    blocks: List[TensorBlock] = []

    for i in range(len(keys)):
        key = keys[i]
        blocks_to_join: List[TensorBlock] = []
        for tensor in tensors:
            blocks_to_join.append(tensor.block(key))

        if axis == "samples":
            blocks.append(_join_block_samples(blocks_to_join, "join", add_dimension))
        else:
            blocks.append(_join_block_properties(blocks_to_join, "join", add_dimension))

    return TensorMap(keys=keys, blocks=blocks)


@torch_jit_script
def join_blocks(
    blocks: List[TensorBlock],
    axis: str,
    add_dimension: Optional[str] = None,
) -> TensorBlock:
    """Join a sequence of :py:class:`TensorBlock` along an axis.

    The ``axis`` parameter specifies the type of joining: with ``axis='properties'`` the
    ``blocks`` will be joined along the ``properties`` and for ``axis='samples'`` they
    will be joined along the ``samples``.

    :param tensors: sequence of :py:class:`TensorMap` to join
    :param axis: Along which axis the blocks should be joined. This can be either
        ``'properties'`` or ``'samples'``.
    :param add_dimension: Add an the extra dimension to the joined labels with the given
        name. The dimension forms the last dimension of the joined labels.
    :return: The joined :py:class:`TensorBlock` with more properties or samples than the
        inputs.

    .. seealso::

        The examples for :py:func:`join`.
    """
    if not torch_jit_is_scripting():
        if not isinstance(blocks, (list, tuple)):
            raise TypeError(f"`blocks` must be a list or a tuple, not {type(blocks)}")

        for block in blocks:
            if not isinstance_metatensor(block, "TensorBlock"):
                raise TypeError(
                    "`blocks` elements must be metatensor TensorBlock, "
                    f"not {type(block)}"
                )

    if len(blocks) == 0:
        raise ValueError("provide at least one `TensorBlock` for joining")

    if axis not in ("samples", "properties"):
        raise ValueError(
            "Only `'properties'` or `'samples'` are valid values for the `axis` "
            "parameter"
        )

    if axis == "samples":
        names_list = [block.samples.names for block in blocks]
        names_list_flattened: List[str] = []
        for names in names_list:
            names_list_flattened += names

        unique_names = _unique_str(names_list_flattened)
        length_equal = [len(unique_names) == len(names) for names in names_list]
        samples_names_are_same = sum(length_equal) == len(length_equal)

        if not samples_names_are_same:
            # It's fine to lose metadata for the properties, less so for the samples!
            raise ValueError(
                "Different blocks have different sample names in `join_blocks`. "
                "Joining along samples with different sample names will lose "
                "information and is not supported."
            )
        return _join_block_samples(blocks, "join_blocks", add_dimension)
    else:
        return _join_block_properties(blocks, "join_blocks", add_dimension)
