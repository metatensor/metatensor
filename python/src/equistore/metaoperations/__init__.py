"""
The Python API for equistore also provides functions which operate on
tensor metadata stored in :py:class:`equistore.Labels` objects.
"""
from .equal import equal  # noqa
from .intersection import intersection  # noqa
from .unique import unique, unique_block  # noqa


__all__ = [
    "equal",
    "intersection",
    "unique",
    "unique_block",
]
