import sys


try:
    import metatensor_learn
except ImportError as e:
    raise ImportError(
        "metatensor-learn is required to use the metatensor.learn module. "
        "Please install it with `pip install metatensor-learn` or using "
        "your favorite Python package manager."
    ) from e


# metatensor.learn is registered as an alias in metatensor_learn's __init__.py
assert sys.modules["metatensor.learn"] is metatensor_learn
