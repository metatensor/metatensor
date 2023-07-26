"""
Manipulating TensorMap dimensions
=================================

Functions for manipulating dimensions of a :py:class:`equistore.TensorMap` (i.e.
changing the columns of the :py:class:`equistore.Labels` within).

.. autofunction:: equistore.append_dimension

.. autofunction:: equistore.insert_dimension

.. autofunction:: equistore.remove_dimension

.. autofunction:: equistore.rename_dimension
"""
import numpy as np

from equistore.core import TensorBlock, TensorMap


def _check_axis(axis: str):
    if axis not in ["keys", "samples", "properties"]:
        raise ValueError(
            f"{axis} is not a valid axis. Choose from 'keys', 'samples' or 'properties'"
        )


def append_dimension(
    tensor: TensorMap,
    axis: str,
    name: str,
    values: np.ndarray,
) -> TensorMap:
    """Append a :py:class:`equistore.Labels` dimension along the given axis.

    For ``axis=="samples"`` the new dimension is `not` appended to gradients.

    :param tensor: the input :py:class:`TensorMap`.
    :param axis: axis for which the ``name`` should be appended. Allowed are ``"keys"``,
                 ``"properties"`` or ``"samples"``.
    :param name: name of the dimension be appended
    :param values: values of the dimension to be appended


    :raises ValueError: if ``axis`` is a not valid value

    :return: a new :py:class:`equistore.TensorMap` with appended labels dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import equistore
    >>> values = np.array([[1, 2], [3, 4]])
    >>> block = equistore.block_from_array(values)
    >>> keys = equistore.Labels(["foo"], np.array([[0]]))
    >>> tensor = equistore.TensorMap(keys=keys, blocks=[block])
    >>> tensor
    TensorMap with 1 blocks
    keys: foo
           0
    >>> equistore.append_dimension(
    ...     tensor, name="bar", values=np.array([1]), axis="keys"
    ... )
    TensorMap with 1 blocks
    keys: foo  bar
           0    1
    """
    kwargs = dict(tensor=tensor, axis=axis, name=name, values=values)
    if axis == "keys":
        index = len(tensor.keys.names)
        return insert_dimension(index=index, **kwargs)
    elif axis == "samples":
        index = len(tensor.sample_names)
        return insert_dimension(index=index, **kwargs)
    elif axis == "properties":
        index = len(tensor.property_names)
        return insert_dimension(index=index, **kwargs)
    else:
        raise ValueError(
            f"{axis} is not a valid axis. Choose from 'keys', 'samples' or 'properties'"
        )


def insert_dimension(
    tensor: TensorMap,
    axis: str,
    index: int,
    name: str,
    values: np.ndarray,
) -> TensorMap:
    """Insert a :py:class:`equistore.Labels` dimension along the given axis before the
    given index.

    For ``axis=="samples"`` a new dimension is `not` appended to gradients.

    :param tensor: the input :py:class:`TensorMap`.
    :param axis: axis for which the ``name`` should be inserted. Allowed are ``"keys"``,
                 ``"properties"`` or ``"samples"``.
    :param index: index before the new dimension is inserted.
    :param name: the name to be inserted
    :param values: values to be inserted

    :raises ValueError: if ``axis`` is a not valid value

    :return: a new :py:class:`equistore.TensorMap` with inserted labels dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import equistore
    >>> values = np.array([[1, 2], [3, 4]])
    >>> block = equistore.block_from_array(values)
    >>> keys = equistore.Labels(["foo"], np.array([[0]]))
    >>> tensor = equistore.TensorMap(keys=keys, blocks=[block])
    >>> tensor
    TensorMap with 1 blocks
    keys: foo
           0
    >>> equistore.insert_dimension(
    ...     tensor, index=0, name="bar", values=np.array([1]), axis="keys"
    ... )
    TensorMap with 1 blocks
    keys: bar  foo
           1    0
    """
    _check_axis(axis)

    keys = tensor.keys
    if axis == "keys":
        keys = keys.insert(index=index, name=name, values=values)

    blocks = []
    for block in tensor:
        samples = block.samples
        properties = block.properties

        if axis == "samples":
            samples = samples.insert(index=index, name=name, values=values)
        elif axis == "properties":
            properties = properties.insert(index=index, name=name, values=values)

        new_block = TensorBlock(
            values=block.values,
            samples=samples,
            components=block.components,
            properties=properties,
        )

        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=properties,
                ),
            )

        blocks.append(new_block)

    return TensorMap(keys=keys, blocks=blocks)


def remove_dimension(tensor: TensorMap, axis: str, name: str) -> TensorMap:
    """Remove a :py:class:`equistore.Labels` dimension along the given axis.

    Removal can only be performed if the resulting :py:class:`equistore.Labels`
    instance will be unique.

    For ``axis=="samples"`` the dimension is not removed from gradients.

    :param tensor: the input :py:class:`TensorMap`.
    :param axis: axis for which ``name`` should be removed. Allowed are ``"keys"``,
                 ``"properties"`` or ``"samples"``.
    :param name: the :py:class:`equistore.Labels` name to be removed

    :raises ValueError: if ``axis`` is a not valid value
    :raises ValueError: if the only dimension should be removed
    :raises ValueError: if name is not part of the axis

    :return: a new :py:class:`equistore.TensorMap` with removed labels dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import equistore
    >>> values = np.array([[1, 2], [3, 4]])
    >>> block = equistore.block_from_array(values)
    >>> keys = equistore.Labels(["key", "extra"], np.array([[0, 0]]))
    >>> tensor = equistore.TensorMap(keys=keys, blocks=[block])
    >>> tensor
    TensorMap with 1 blocks
    keys: key  extra
           0     0
    >>> equistore.remove_dimension(tensor, axis="keys", name="extra")
    TensorMap with 1 blocks
    keys: key
           0

    Removing a dimension can only be performed if the resulting :py:class:`Labels` will
    contain unique entries.

    >>> from equistore import EquistoreError
    >>> block = equistore.block_from_array(values)
    >>> keys = equistore.Labels(["key", "extra"], np.array([[0, 0], [0, 1]]))
    >>> tensor = equistore.TensorMap(keys=keys, blocks=[block.copy(), block.copy()])
    >>> tensor
    TensorMap with 2 blocks
    keys: key  extra
           0     0
           0     1
    >>> try:
    ...     equistore.remove_dimension(tensor, axis="keys", name="extra")
    ... except EquistoreError as e:
    ...     print(e)
    ...
    invalid parameter: can not have the same label value multiple time: [0] is already present at position 0
    """  # noqa E501
    _check_axis(axis)

    keys = tensor.keys
    if axis == "keys":
        keys = keys.remove(name=name)

    blocks = []
    for block in tensor:
        samples = block.samples
        properties = block.properties

        if axis == "samples":
            samples = samples.remove(name)
        elif axis == "properties":
            properties = properties.remove(name)

        new_block = TensorBlock(
            values=block.values,
            samples=samples,
            components=block.components,
            properties=properties,
        )

        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=properties,
                ),
            )

        blocks.append(new_block)

    return TensorMap(keys=keys, blocks=blocks)


def rename_dimension(tensor: TensorMap, axis: str, old: str, new: str) -> TensorMap:
    """Rename a :py:class:`equistore.Labels` dimension name for a given axis.

    :param tensor: the input :py:class:`TensorMap`.
    :param axis: axis for which the names should be appended. Allowed are ``"keys"``,
                 ``"properties"`` or ``"samples"``.
    :param old: name to be replaced
    :param new: name after the replacement

    :raises ValueError: if ``axis`` is a not valid value

    :return: a `new` :py:class:`equistore.TensorMap` with renamed labels dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import equistore
    >>> values = np.array([[1, 2], [3, 4]])
    >>> block = equistore.block_from_array(values)
    >>> keys = equistore.Labels(["foo"], np.array([[0]]))
    >>> tensor = equistore.TensorMap(keys=keys, blocks=[block])
    >>> tensor
    TensorMap with 1 blocks
    keys: foo
           0
    >>> equistore.rename_dimension(tensor, axis="keys", old="foo", new="bar")
    TensorMap with 1 blocks
    keys: bar
           0

    """
    _check_axis(axis)

    keys = tensor.keys
    if axis == "keys":
        keys = keys.rename(old, new)

    blocks = []
    for block in tensor:
        samples = block.samples
        properties = block.properties

        if axis == "samples":
            samples = samples.rename(old, new)
        elif axis == "properties":
            properties = properties.rename(old, new)

        new_block = TensorBlock(
            values=block.values,
            samples=samples,
            components=block.components,
            properties=properties,
        )

        for parameter, gradient in block.gradients():
            gradient_samples = gradient.samples
            if axis == "samples" and old in gradient_samples.names:
                gradient_samples = gradient_samples.rename(old, new)

            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient_samples,
                    components=gradient.components,
                    properties=properties,
                ),
            )

        blocks.append(new_block)

    return TensorMap(keys=keys, blocks=blocks)
