import sys


try:
    import metatensor_operations
except ImportError as e:
    raise ImportError(
        "metatensor-operations is required to use the metatensor.operations module. "
        "Please install it with `pip install metatensor-operations` or using "
        "your favorite Python package manager."
    ) from e


# metatensor.operations is registered as an alias in metatensor_operations's __init__.py
assert sys.modules["metatensor.operations"] is metatensor_operations
