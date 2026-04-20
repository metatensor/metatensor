import sys


try:
    import metatensor_torch
except ImportError as e:
    raise ImportError(
        "metatensor-torch is required to use the metatensor.torch module. "
        "Please install it with `pip install metatensor-torch` or using "
        "your favorite Python package manager."
    ) from e

# metatensor.torch is registered as an alias in metatensor_torch's __init__.py
assert sys.modules["metatensor.torch"] is metatensor_torch
