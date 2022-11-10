import numpy as np

from equistore import TensorBlock


def compare_blocks(block1: TensorBlock, block2: TensorBlock, rtol=1e-13):
    """
    Compare two ``TensorBlock``s to see if they are the same.
    This works only for numpy array.
    Returns a dictionary in which the keys are:
    global : bool -> are the two blocks globally equal ?
    values, properties, ... : bool -> one for each property of the block

    if the gradient is present a nested dictionary
    with the same structure is present in the returned dictionary.
    """
    result = {}
    result["values"] = np.allclose(block1.values, block2.values, rtol=rtol)
    result["samples"] = np.all(block1.samples == block2.samples)
    if len(block1.properties) == len(block2.properties):
        result["properties"] = np.all(
            [np.all(p1 == p2) for p1, p2 in zip(block1.properties, block2.properties)]
        )
    else:
        result["properties"] = False

    result["components"] = True
    if len(block1.components) > 0:
        for icomp, bc in enumerate(block1.components):
            result["components"] = result["components"] and np.all(
                bc == block2.components[icomp]
            )
    if len(block1.components) == 0 and len(block2.components) > len(block1.components):
        result["components"] = False

    result["gradients"] = {}
    if len(block1.gradients_list()) > 0 and len(block1.gradients_list()) == len(
        block2.gradients_list()
    ):
        result["gradients"]["general"] = True
        for parameter, gradient1 in block1.gradients():
            gradient2 = block2.gradient(parameter)
            result["gradients"][parameter] = {}
            result["gradients"][parameter]["samples"] = np.all(
                gradient1.samples == gradient2.samples
            )

            if len(gradient1.components) > 0 and len(gradient1.components) == len(
                gradient2.components
            ):
                result["gradients"][parameter]["components"] = np.all(
                    [
                        np.all(c1 == c2)
                        for c1, c2 in zip(gradient1.components, gradient2.components)
                    ]
                )
            elif len(gradient1.components) != len(gradient2.components):
                result["gradients"][parameter]["components"] = False
            else:
                result["gradients"][parameter]["components"] = True

            if len(gradient1.properties) > 0 and len(gradient1.properties) == len(
                gradient2.properties
            ):
                result["gradients"][parameter]["properties"] = np.all(
                    [
                        np.all(p1 == p2)
                        for p1, p2 in zip(gradient1.properties, gradient2.properties)
                    ]
                )
            elif len(gradient1.properties) != len(gradient2.properties):
                result["gradients"][parameter]["components"] = False
            else:
                result["gradients"][parameter]["properties"] = True

            result["gradients"][parameter]["data"] = np.allclose(
                gradient1.data, gradient2.data, rtol=1e-13
            )

            result["gradients"][parameter]["general"] = (
                result["gradients"][parameter]["samples"]
                and result["gradients"][parameter]["components"]
                and result["gradients"][parameter]["properties"]
                and result["gradients"][parameter]["data"]
            )
            result["gradients"]["general"] = (
                result["gradients"]["general"]
                and result["gradients"][parameter]["general"]
            )

    elif len(block1.gradients_list()) != len(block2.gradients_list()):
        result["gradients"]["general"] = False
    else:
        result["gradients"]["general"] = True

    result["general"] = (
        result["values"]
        and result["samples"]
        and result["properties"]
        and result["components"]
        and result["gradients"]["general"]
    )
    return result
