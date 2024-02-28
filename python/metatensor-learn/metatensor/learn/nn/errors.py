"""
Module containing the :py:class:`AbsoluteError` and :py:class:`SquaredError` classes.
"""

from .._backend import TensorMap
from .module_map import ModuleMap


class AbsoluteError(ModuleMap):

    def __init__(self):
        pass

    def __call__(self, A: TensorMap, B: TensorMap) -> TensorMap:
        pass


class SquaredError(ModuleMap):

    def __init__(self):
        pass

    def __call__(self, A: TensorMap, B: TensorMap) -> TensorMap:
        pass
