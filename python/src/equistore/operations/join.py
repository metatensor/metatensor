from typing import List

import numpy as np

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap
from . import _dispatch
from ._utils import _check_blocks, _check_same_gradients_components, _check_same_keys


def join(tensormaps: List[TensorMap], axis: str):
    """Join a sequence of :py:class:`TensorMap` along an axis.

    The ``axis`` parameter specifies the type join.
    For example, if ``axis='properties'`` it will be the joined along the
    properties of the TensorMaps and if ``axis='samples'`` it will be the
    along the samples.

    :param tensormaps: sequence of :py:class:`TensorMap` for join
    :param axis: A string indicating how the tensormaps are stacked.
                 Allowed values are ``'properties'`` or ``'samples'``.

    :return: The stacked :py:class:`TensorMap` which has more properties or
             samples than the input TensorMap.
    """

    if len(tensormaps) < 2 or type(tensormaps) not in [list, tuple]:
        raise ValueError("provide at least two `TensorMap`s for joining")

    for ts_to_join in tensormaps[1:]:
        _check_same_keys(tensormaps[0], ts_to_join, "join")

    blocks = []
    for key in tensormaps[0].keys:
        blocks_to_join = [ts.block(key) for ts in tensormaps]

        if axis == "properties":
            blocks.append(_join_blocks_along_properties(blocks_to_join))
        elif axis == "samples":
            blocks.append(_join_blocks_along_samples(blocks_to_join))
        else:
            raise ValueError(
                "Only `'properties'` or `'samples'` are "
                "valid values for the `axis` parameter."
            )

    return TensorMap(tensormaps[0].keys, blocks)


def _join_labels(labels: List[Labels]) -> Labels:
    """Join a sequence of :py:class:`Labels`

    Possible name and value clashes are resolved by the following strategy:

    1. If the names are the same and the values are unique we keep the names and
       join the values vertically. In example::

        labels_1 = ["n"], [0, 2, 3]
        labels_2 = ["n"], [1, 4, 5]

      will lead to::

        properties = ["n"], [0, 2, 3, 1, 4, 5]

    2. If the names are the same but the values are not unique a new variable
       "tensor" is added to the names. In example::

            properties_1 = ["n"], [0, 2, 3]
            properties_2 = ["n"], [0, 2]

        will lead to::
            properties = ["tensor", "n"], [[0, 0], [0, 2], [0, 3], [1, 0], [1, 2]]

    3. If names of the Labels are different we change the names two to
       ("tensor", "property"). In example::

            properties_1 = ["a"], [0, 2, 3]
            properties_2 = ["b", "c], [[0, 0], [1, 2]]

        will lead to::

            properties = ["tensor", "q"], [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]
    """
    names = [label.names for label in labels]
    values = [label.tolist() for label in labels]
    tensor_values = np.repeat(
        a=np.arange(len(values)), repeats=[len(value) for value in values]
    )

    unique_names = np.unique(names)
    unique_values = np.unique(np.vstack(values), axis=0)

    names_are_same = np.all(
        len(unique_names) == np.array([len(name) for name in names])
    )
    values_are_unique = len(unique_values) == len(np.vstack(values))

    if names_are_same and values_are_unique:
        new_names = names[0]
        new_values = np.vstack(values)
    elif names_are_same and not values_are_unique:
        new_names = ["tensor"] + list(names[0])
        new_values = np.hstack([tensor_values.reshape(-1, 1), np.vstack(values)])
    else:
        new_names = ["tensor", "property"]
        property_values = np.hstack([np.arange(len(value)) for value in values])
        new_values = np.vstack([tensor_values, property_values]).T

    return Labels(names=new_names, values=new_values)


def _join_blocks_along_properties(blocks: List[TensorBlock]) -> TensorBlock:
    """Join a sequence of :py:class:`TensorBlock` along their properties."""
    first_block = blocks[0]

    fname = "_join_blocks_along_properties"
    for block in blocks[1:]:
        _check_blocks(first_block, block, ["components", "samples", "gradients"], fname)
        _check_same_gradients_components(first_block, block, fname)

    properties = _join_labels([block.properties for block in blocks])
    result_block = TensorBlock(
        values=_dispatch.hstack([block.values for block in blocks]),
        samples=first_block.samples,
        components=first_block.components,
        properties=properties,
    )

    for parameter, first_gradient in first_block.gradients():

        # Different blocks might contain different gradient samples
        gradient_sample_values = np.unique(
            np.hstack([block.gradient(parameter).samples for block in blocks])
        ).tolist()
        gradient_samples = Labels(
            names=first_gradient.samples.names, values=np.array(gradient_sample_values)
        )

        gradient_data = np.zeros(
            (len(gradient_samples),)
            + first_gradient.data.shape[1:-1]
            + (len(properties),)
        )

        properties_start = 0
        for block in blocks:
            gradient = block.gradient(parameter)

            # find the correct sample position to put the data
            for i_grad, sample_grad in enumerate(gradient.samples):
                # TODO: once C-API is changed replace by:
                # i_sample = gradient_samples.position(sample_grad)`
                i_sample = np.where(gradient_samples == sample_grad)[0][0]
                gradient_data[
                    i_sample,
                    ...,
                    properties_start : properties_start + len(gradient.properties),
                ] = gradient.data[i_grad]

            properties_start += len(gradient.properties)

        result_block.add_gradient(
            parameter, gradient_data, gradient_samples, first_gradient.components
        )

    return result_block


def _join_blocks_along_samples(blocks: List[TensorBlock]) -> TensorBlock:
    """Join a sequence of :py:class:`TensorBlock` along their samples."""
    first_block = blocks[0]

    fname = "_join_blocks_along_samples"
    for block in blocks[1:]:
        _check_blocks(
            first_block, block, ["components", "properties", "gradients"], fname
        )
        _check_same_gradients_components(first_block, block, fname)

    result_block = TensorBlock(
        values=_dispatch.vstack([block.values for block in blocks]),
        samples=_join_labels([block.samples for block in blocks]),
        components=first_block.components,
        properties=first_block.properties,
    )

    for parameter, first_gradient in first_block.gradients():
        gradient_data = [first_gradient.data]
        gradient_samples = [np.array(first_gradient.samples.tolist())]

        for block in blocks[1:]:
            gradient = block.gradient(parameter)
            gradient_data.append(gradient.data)

            samples = np.array(gradient.samples.tolist())
            # update the "sample" variable in gradient samples
            samples[:, 0] += 1 + gradient_samples[-1][:, 0].max()
            gradient_samples.append(samples)

        gradient_data = np.vstack(gradient_data)
        gradient_samples = np.vstack(gradient_samples)

        gradients_samples = Labels(first_gradient.samples.names, gradient_samples)

        result_block.add_gradient(
            parameter,
            gradient_data,
            gradients_samples,
            first_gradient.components,
        )

    return result_block
