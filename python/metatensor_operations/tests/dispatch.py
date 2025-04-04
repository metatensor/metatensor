import numpy as np
import pytest

import metatensor


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    create_array_functions = [np.array, torch.tensor]
else:
    create_array_functions = [np.array]


all_true_array = [True, True, True]
all_false_array = [True, True, False]


@pytest.mark.parametrize("create_array_function", create_array_functions)
def test_all(create_array_function):
    assert metatensor.operations._dispatch.all(create_array_function(all_true_array))
    assert not metatensor.operations._dispatch.all(
        create_array_function(all_false_array)
    )
