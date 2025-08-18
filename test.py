import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.learn.nn import (
    EquivariantLinear,
    InvariantReLU,
    Sequential,
)


torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

spherical_expansion = mts.load("spherical-expansion.mts")

# metatensor-learn modules currently do not support TensorMaps with gradients
spherical_expansion = mts.remove_gradients(spherical_expansion)
print(spherical_expansion)
print("\nNumber of blocks in the spherical expansion:", len(spherical_expansion))
