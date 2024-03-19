from typing import List, Union

from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    check_isinstance,
    torch_jit_is_scripting,
    torch_jit_script,
)
from ._utils import _check_same_keys_raise
from .manipulate_dimension import remove_dimension


def _disjoint_tensor_labels(tensors: List[TensorMap], axis: str) -> bool:
    """Checks if all labels in a list of TensorMaps are disjoint.

    We have to perform a check from all tensors to all others to ensure it
    they are "fully" disjoint.
    """
    for i_tensor, first_tensor in enumerate(tensors[:-1]):
        for second_tensor in tensors[i_tensor + 1 :]:
            for key, first_block in first_tensor.items():
                second_block = second_tensor.block(key)
                if axis == "samples":
                    first_labels = first_block.samples
                    second_labels = second_block.samples
                elif axis == "properties":
                    first_labels = first_block.properties
                    second_labels = second_block.properties
                else:
                    raise ValueError(
                        "Only `'properties'` or `'samples'` are "
                        "valid values for the `axis` parameter."
                    )

                if len(first_labels.intersection(second_labels)):
                    return False

    return True


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


@torch_jit_script
def join(
    tensors: List[TensorMap],
    axis: str,
    different_keys: str = "error",
    sort_samples: bool = False,
    remove_tensor_name: bool = False,
) -> TensorMap:
    """Join a sequence of :py:class:`TensorMap` with the same blocks along an axis.

    The ``axis`` parameter specifies the type of joining. For example, if
    ``axis='properties'`` the tensor maps in `tensors` will be joined along the
    `properties` dimension and for ``axis='samples'`` they will be the along the
    `samples` dimension.

    :param tensors:
        sequence of :py:class:`TensorMap` for join
    :param axis:
        A string indicating how the tensormaps are stacked. Allowed
        values are ``'properties'`` or ``'samples'``.
    :param different_keys: Method to handle different keys between the tensors. For
        ``"error"`` keys in all tensors have to be the same. For ``"intersection"`` only
        blocks present in all tensors will be taken into account. For ``"union"``
        missing keys will be treated like if they where associated with an empty block.
    :param sort_samples: whether to sort the samples of the merged ``tensors`` or keep
        them in their original order.
    :param remove_tensor_name:
        Remove the extra ``tensor`` dimension from labels if possible. See examples
        above for the case where this is applicable.
    :return tensor_joined:
        The stacked :py:class:`TensorMap` with more properties or samples
        than the input TensorMap.

    Examples
    --------
    Possible clashes of the meta data like ``samples``/``properties`` will be resolved
    by one of the three following strategies:

    1. If Labels names are the same, the values are unique and
       ``remove_tensor_name=True`` we keep the names and join the values

       >>> import numpy as np
       >>> import metatensor
       >>> from metatensor import Labels, TensorBlock, TensorMap

       >>> values = np.array([[1.1, 2.1, 3.1]])
       >>> samples = Labels("sample", np.array([[0]]))

       Define two disjoint :py:class:`Labels`.

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

       joining along the properties leads

       >>> joined_tensor = metatensor.join(
       ...     [tensor_1, tensor_2], axis="properties", remove_tensor_name=True
       ... )
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

       If ``remove_tensor_name=False`` There will be an extra dimension ``tensor``
       added

       >>> joined_tensor = metatensor.join(
       ...     [tensor_1, tensor_2], axis="properties", remove_tensor_name=False
       ... )
       >>> joined_tensor[0].properties
       Labels(
           tensor  n
             0     0
             0     2
             0     3
             1     1
             1     4
             1     5
       )

    2. If Labels names are the same but the values are not unique, a new dimension
       ``"tensor"`` is added to the names.

       >>> properties_3 = Labels("n", np.array([[0], [2], [3]]))

       ``properties_3`` has the same name and also shares values with ``properties_1``
       as defined above.

       >>> block_3 = TensorBlock(
       ...     values=values,
       ...     samples=Labels.single(),
       ...     components=[],
       ...     properties=properties_3,
       ... )
       >>> tensor_3 = TensorMap(keys=Labels.single(), blocks=[block_3])

       joining along properties leads to

       >>> joined_tensor = metatensor.join([tensor_1, tensor_3], axis="properties")
       >>> joined_tensor[0].properties
       Labels(
           tensor  n
             0     0
             0     2
             0     3
             1     0
             1     2
             1     3
       )

    3. If Labels names are different we change the names to ("tensor", "property"). This
       case is only supposed to happen when joining in the property dimension, hence the
       choice of names:

       >>> properties_4 = Labels(["a", "b"], np.array([[0, 0], [1, 2], [1, 3]]))

       ``properties_4`` has the different names compared to ``properties_1``
       defined above.

       >>> block_4 = TensorBlock(
       ...     values=values,
       ...     samples=Labels.single(),
       ...     components=[],
       ...     properties=properties_4,
       ... )
       >>> tensor_4 = TensorMap(keys=Labels.single(), blocks=[block_4])

       joining along properties leads to

        >>> joined_tensor = metatensor.join([tensor_1, tensor_4], axis="properties")
        >>> joined_tensor[0].properties
        Labels(
            tensor  property
              0        0
              0        1
              0        2
              1        0
              1        1
              1        2
        )
    """
    if not torch_jit_is_scripting():
        if not isinstance(tensors, (list, tuple)):
            raise TypeError(f"`tensor` must be a list or a tuple, not {type(tensors)}")

        for tensor in tensors:
            if not check_isinstance(tensor, TensorMap):
                raise TypeError(
                    "`tensors` elements must be metatensor TensorMap, "
                    f"not {type(tensor)}"
                )

    if len(tensors) < 1:
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
    else:
        names_list = [tensor.property_names for tensor in tensors]

    names_list_flattened: List[str] = []
    for names in names_list:
        names_list_flattened += names

    unique_names = _unique_str(names_list_flattened)
    length_equal = [len(unique_names) == len(names) for names in names_list]
    names_are_same = sum(length_equal) == len(length_equal)

    # It's fine to lose metadata on the property axis, less so on the sample axis!
    if axis == "samples" and not names_are_same:
        raise ValueError(
            "Sample names are not the same! Joining along samples with different "
            "sample names will loose information and is not supported."
        )

    keys = tensors[0].keys
    n_tensors = len(tensors)
    n_keys_dimensions = 1 + keys.values.shape[1]
    new_keys_values = _dispatch.empty_like(
        array=keys.values,
        shape=[n_tensors, keys.values.shape[0], n_keys_dimensions],
    )

    for i, tensor in enumerate(tensors):
        for j, value in enumerate(tensor.keys.values):
            new_keys_values[i, j, 0] = i
            new_keys_values[i, j, 1:] = value

    keys = Labels(
        names=["tensor"] + keys.names,
        values=new_keys_values.reshape(-1, n_keys_dimensions),
    )

    blocks: List[TensorBlock] = []
    for tensor in tensors:
        for block in tensor.blocks():
            # We would already raised an error if `axis == "samples"`. Therefore, we can
            # neglect the check for `axis == "properties"`.
            if names_are_same:
                properties = block.properties
            else:
                properties = Labels(
                    names=["property"],
                    values=_dispatch.int_array_like(
                        list(range(len(block.properties))), block.properties.values
                    ).reshape(-1, 1),
                )

            new_block = TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            for parameter, gradient in block.gradients():
                if len(gradient.gradients_list()) != 0:
                    raise NotImplementedError(
                        "gradients of gradients are not supported"
                    )

                new_block.add_gradient(
                    parameter=parameter,
                    gradient=TensorBlock(
                        values=gradient.values,
                        samples=gradient.samples,
                        components=gradient.components,
                        properties=new_block.properties,
                    ),
                )

            blocks.append(new_block)

    tensor = TensorMap(keys=keys, blocks=blocks)

    if axis == "samples":
        tensor_joined = tensor.keys_to_samples("tensor", sort_samples=sort_samples)
    else:
        tensor_joined = tensor.keys_to_properties("tensor", sort_samples=sort_samples)

    if remove_tensor_name and _disjoint_tensor_labels(tensors, axis):
        return remove_dimension(tensor_joined, name="tensor", axis=axis)
    else:
        return tensor_joined
