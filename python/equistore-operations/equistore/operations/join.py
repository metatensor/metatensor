import functools
import operator
from typing import List

import numpy as np

from equistore.core import Labels, TensorBlock, TensorMap

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

                if len(first_labels.intersection(second_labels)):
                    return False

    return True


def join(
    tensors: List[TensorMap], axis: str, remove_tensor_name: bool = False
) -> TensorMap:
    """Join a sequence of :py:class:`TensorMap` along an axis.

    The ``axis`` parameter specifies the type of joining. For example, if
    ``axis='properties'`` the tensor maps in `tensors` will be joined along the
    `properties` dimension and for ``axis='samples'`` they will be the along the
    `samples` dimension.

    :param tensors:
        sequence of :py:class:`TensorMap` for join
    :param axis:
        A string indicating how the tensormaps are stacked. Allowed
        values are ``'properties'`` or ``'samples'``.
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
       >>> import equistore
       >>> from equistore import Labels, TensorBlock, TensorMap

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

       >>> joined_tensor = equistore.join(
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

       >>> joined_tensor = equistore.join(
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

       >>> joined_tensor = equistore.join([tensor_1, tensor_3], axis="properties")
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

        >>> joined_tensor = equistore.join([tensor_1, tensor_4], axis="properties")
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

    if not isinstance(tensors, (list, tuple)):
        raise TypeError(
            "the `TensorMap`s to join must be provided as a list or a tuple"
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

    for ts_to_join in tensors[1:]:
        _check_same_keys_raise(tensors[0], ts_to_join, "join")

    # Deduce if sample/property names are the same in all tensors.
    # If this is not the case we have to change unify the corresponding labels later.
    if axis == "samples":
        names_list = [tensor.sample_names for tensor in tensors]
    else:
        names_list = [tensor.property_names for tensor in tensors]

    # We use functools to flatten a list of sublists::
    #
    #   [('a', 'b', 'c'), ('a', 'b')] -> ['a', 'b', 'c', 'a', 'b']
    #
    # A nested list with sublist of different shapes can not be handled by np.unique.
    unique_names = np.unique(functools.reduce(operator.concat, names_list))

    # Label names are unique: We can do an equal check only checking the lengths.
    names_are_same = np.all(
        len(unique_names) == np.array([len(names) for names in names_list])
    )

    # It's fine to lose metadata on the property axis, less so on the sample axis!
    if axis == "samples" and not names_are_same:
        raise ValueError(
            "Sample names are not the same! Joining along samples with different "
            "sample names will loose information and is not supported."
        )

    keys_names = ["tensor"] + tensors[0].keys.names
    keys_values = []
    blocks = []

    for i, tensor in enumerate(tensors):
        keys_values += [[i] + value.tolist() for value in tensor.keys.values]

        for block in tensor:
            # We would already raised an error if `axis == "samples"`. Therefore, we can
            # neglect the check for `axis == "properties"`.
            if names_are_same:
                properties = block.properties
            else:
                properties = Labels.range("property", len(block.properties))

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

    keys = Labels(names=keys_names, values=np.array(keys_values))
    tensor = TensorMap(keys=keys, blocks=blocks)

    if axis == "samples":
        tensor_joined = tensor.keys_to_samples("tensor")
    else:
        tensor_joined = tensor.keys_to_properties("tensor")

    if remove_tensor_name and _disjoint_tensor_labels(tensors, axis):
        return remove_dimension(tensor_joined, name="tensor", axis=axis)
    else:
        return tensor_joined
