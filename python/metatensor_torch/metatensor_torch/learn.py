import importlib
import sys
from typing import Union

import torch

from . import Labels, LabelsEntry, TensorBlock, TensorMap


# ==================================================================================== #
# see operations.py for an explanation of what's going on here.                        #
# ==================================================================================== #

try:
    import metatensor_learn
except ImportError as e:
    raise ImportError(
        "metatensor-learn is required to use the metatensor.torch.learn module. "
        "Please install it with `pip install metatensor-learn` or using "
        "your favorite Python package manager."
    ) from e


# Step 1: create the `_backend` module as an empty module
spec = importlib.util.spec_from_loader(
    "metatensor_torch.learn._backend",
    loader=None,
)
backend_module = importlib.util.module_from_spec(spec)
# This module only exposes a handful of things, defined here. Any changes here MUST also
# be made to the `metatensor/learn/_backend.py` file, which is used in non
# TorchScript mode.
backend_module.__dict__["Labels"] = Labels
backend_module.__dict__["LabelsEntry"] = LabelsEntry
backend_module.__dict__["TensorBlock"] = TensorBlock
backend_module.__dict__["TensorMap"] = TensorMap
backend_module.__dict__["torch_jit_is_scripting"] = torch.jit.is_scripting


def isinstance_metatensor(value: Union[Labels, TensorBlock, TensorMap], typename: str):
    assert typename in ("Labels", "TensorBlock", "TensorMap")

    if torch.jit.is_scripting():
        if typename == "Labels":
            return isinstance(value, Labels)
        elif typename == "TensorBlock":
            return isinstance(value, TensorBlock)
        elif typename == "TensorMap":
            return isinstance(value, TensorMap)

    # For custom classes (TensorMap, …), the `values` is an instance of
    # `torch.ScriptObject` and not the class itself, so we use `_type`
    # to get the type name
    if isinstance(value, torch.ScriptObject):
        qualified_name = value._type().qualified_name()
        if qualified_name.startswith("__torch__.torch.classes.metatensor"):
            return value._type().name() == typename

    return False


backend_module.__dict__["isinstance_metatensor"] = isinstance_metatensor

# register the module in sys.modules, so future import find it directly
sys.modules[spec.name] = backend_module


# Step 2: create a module named `metatensor_torch.learn` (like the one that will be
# created by importing the current file), but using code from `metatensor.learn`
spec = importlib.util.spec_from_file_location(
    # create a module with this name
    "metatensor_torch.learn",
    # using the code from there
    metatensor_learn.__file__,
)

module = importlib.util.module_from_spec(spec)
# override `metatensor_torch.learn` (the module associated with the current file)
# with the newly created module
sys.modules[spec.name] = module
spec.loader.exec_module(module)

module._backend = backend_module
