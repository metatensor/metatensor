from equistore import TensorBlock, TensorMap

from . import _dispatch


def normalize_by_sample(tensor: TensorMap) -> TensorMap:
    """Normalize each sample (row) in the blocks of the input TensorMap.

    TODO: more information on this function
    """
    # TODO: allow to normalize the tensor map in place (with a parameter to pick
    # between in-place & copying). This will not work with pytorch autograd, but
    # might be better than allocating new blocks all the time.

    blocks = []
    for _, block in tensor:
        if len(block.components) != 0:
            raise ValueError(
                "normalization of equivariant tensors is not yet implemented"
            )

        values = block.values
        norm = _dispatch.norm(values, axis=-1)

        normalized_values = values / norm[..., None]

        new_block = TensorBlock(
            values=normalized_values,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter, gradient in block.gradients():
            if parameter != "positions":
                raise ValueError(
                    f"normalization of gradients w.r.t. '{parameter}' is "
                    "not yet implemented"
                )

            gradient_data = (
                gradient.data / norm[gradient.samples["sample"], None, ..., None]
            )

            # gradient of x_i = X_i / N_i is given by
            # 1 / N_i \grad X_i - x_i [x_i @ 1 / N_i \grad X_i]
            for sample_i, (sample, _, _) in enumerate(gradient.samples):
                dot = gradient_data[sample_i] @ normalized_values[sample].T

                gradient_data[sample_i, 0, :] -= dot[0] * normalized_values[sample, :]
                gradient_data[sample_i, 1, :] -= dot[1] * normalized_values[sample, :]
                gradient_data[sample_i, 2, :] -= dot[2] * normalized_values[sample, :]

            new_block.add_gradient(
                parameter,
                gradient_data,
                gradient.samples,
                gradient.components,
            )

        blocks.append(new_block)

    return TensorMap(tensor.keys, blocks)
