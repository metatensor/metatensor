import importlib
import os
import sys
from typing import Union

import torch

from . import Labels, TensorBlock, TensorMap


#                       CAREFUL ADVENTURER, HERE BE DRAGONS!
#
#                                         \||/
#                                         |  @___oo
#                               /\  /\   / (__,,,,|
#                              ) /^\) ^\/ _)
#                              )   /^\/   _)
#                              )   _ /  / _)
#                          /\  )/\/ ||  | )_)
#                         <  >      |(,,) )__)
#                          ||      /    \)___)\
#                          | \____(      )___) )___
#                           \______(_______;;; __;;;
#
#
# This module tries to re-use code from `metatensor-operations` to expose TorchScript
# compatible functions. To achieve this we need two things:
#  - the code needs to use TorchScript compatible operations only
#  - the type annotation of the functions have to refer to classes TorchScript knows
#    about
#
# To achieve this, we import the modules in a special mode with `importlib`, first
# creating the `metatensor_torch.operations._backend` module, containing all metatensor
# classes plus a `TORCH_SCRIPT_MODE` constant to change the code between Python and
# TorchScript modes.
#
# We then import the code from `metatensor-operations` into a custom module
# `metatensor_torch.operations`. When code in this module tries to `from ._backend
# import ...`, the import is resolved to the values defined in the step above. This
# allows to have the right type annotation on the functions.
#
# Overall, the same code is used to define two versions of each function: one will be
# used in `metatensor`, and one in `metatensor_torch`.

try:
    import metatensor_operations
except ImportError as e:
    raise ImportError(
        "metatensor-operations is required to use the metatensor.torch.operations "
        "module. Please install it with `pip install metatensor-operations` or using "
        "your favorite Python package manager."
    ) from e


# Step 1: create the `_backend` module as an empty module
spec = importlib.util.spec_from_loader(
    "metatensor_torch.operations._backend",
    loader=None,
)
backend_module = importlib.util.module_from_spec(spec)
# This module only exposes a handful of things, defined here. Any changes here MUST also
# be made to the `metatensor/operations/_backend.py` file, which is used in non
# TorchScript mode.
backend_module.__dict__["Labels"] = Labels
backend_module.__dict__["TensorBlock"] = TensorBlock
backend_module.__dict__["TensorMap"] = TensorMap
backend_module.__dict__["Array"] = torch.Tensor
backend_module.__dict__["LabelsValues"] = torch.Tensor

if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") != "0":
    # disable TorchScript compilation when importing the code with
    # METATENSOR_IMPORT_FOR_SPHINX. This is required since the classes
    # definition in this case are not compatible with TorchScript
    backend_module.__dict__["torch_jit_is_scripting"] = lambda: False
    backend_module.__dict__["torch_jit_script"] = lambda u: u
else:
    backend_module.__dict__["torch_jit_is_scripting"] = torch.jit.is_scripting
    backend_module.__dict__["torch_jit_script"] = torch.jit.script

backend_module.__dict__["torch_jit_annotate"] = torch.jit.annotate


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


# Step 2: create a module named `metatensor_torch.operations` (like the one that will be
# created by importing the current file), but using code from `metatensor.operations`
spec = importlib.util.spec_from_file_location(
    # create a module with this name
    "metatensor_torch.operations",
    # using the code from there
    metatensor_operations.__file__,
)

module = importlib.util.module_from_spec(spec)
# override `metatensor_torch.operations` (the module associated with the current file)
# with the newly created module
sys.modules[spec.name] = module
spec.loader.exec_module(module)

module._backend = backend_module
