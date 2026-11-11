import numpy as np
import pytest

import metatensor as mts


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    create_array_functions = [np.array, torch.tensor]
else:
    create_array_functions = [np.array]

try:
    import jax.numpy as jnp

    create_array_functions.append(jnp.array)
except ImportError:
    pass


all_true_array = [True, True, True]
all_false_array = [True, True, False]


@pytest.mark.parametrize("create_array_function", create_array_functions)
def test_all(create_array_function):
    assert mts.operations._dispatch.all(create_array_function(all_true_array))
    assert not mts.operations._dispatch.all(create_array_function(all_false_array))


@pytest.mark.parametrize("create_array_function", create_array_functions)
def test_norm(create_array_function):
    array = create_array_function([[3.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(mts.operations._dispatch.norm(array), np.sqrt(125.0))
    np.testing.assert_allclose(
        mts.operations._dispatch.norm(array, axis=1), [5.0, 10.0]
    )
