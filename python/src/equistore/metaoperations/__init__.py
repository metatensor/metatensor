"""
The Python API for equistore also provides functions which operate on
tensor metadata stored in :py:class:`equistore.Labels` objects.
"""
from .unique import unique, unique_block  # noqa


__all__ = [
    "unique",
    "unique_block",
]
