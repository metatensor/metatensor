import importlib
import sys
from typing import Union

import torch

import metatensor.learn
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap


# ==================================================================================== #
# see operations.py for an explanation of what's going on here.                        #
# ==================================================================================== #


# Step 1: create the `_backend` module as an empty module
spec = importlib.util.spec_from_loader(
    "metatensor.torch.learn._backend",
    loader=None,
)
module = importlib.util.module_from_spec(spec)
# This module only exposes a handful of things, defined here. Any changes here MUST also
# be made to the `metatensor/learn/_backend.py` file, which is used in non
# TorchScript mode.
module.__dict__["Labels"] = Labels
module.__dict__["LabelsEntry"] = LabelsEntry
module.__dict__["TensorBlock"] = TensorBlock
module.__dict__["TensorMap"] = TensorMap
module.__dict__["torch_jit_is_scripting"] = torch.jit.is_scripting


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


module.__dict__["isinstance_metatensor"] = isinstance_metatensor

# register the module in sys.modules, so future import find it directly
sys.modules[spec.name] = module


# Step 2: create a module named `metatensor.torch.learn` (like the one that will be
# created by importing the current file), but using code from `metatensor.learn`
spec = importlib.util.spec_from_file_location(
    # create a module with this name
    "metatensor.torch.learn",
    # using the code from there
    metatensor.learn.__file__,
)

module = importlib.util.module_from_spec(spec)
# override `metatensor.torch.learn` (the module associated with the current file)
# with the newly created module
sys.modules[spec.name] = module
spec.loader.exec_module(module)
