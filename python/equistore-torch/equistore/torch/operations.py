import importlib
import sys

import torch

import equistore.operations
from equistore.torch import Labels, TensorBlock, TensorMap


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
# This module tries to re-use code from `equistore-operations` to expose TorchScript
# compatible functions. To achieve this we need two things:
#  - the code needs to use TorchScript compatible operations only
#  - the type annotation of the functions have to refer to classes TorchScript knows
#    about
#
# To achieve this, we import the modules in a special mode with `importlib`, first
# creating the `equistore.torch.operations._classes` module, containing all equistore
# classes plus a `TORCH_SCRIPT_MODE` constant to change the code between Python and
# TorchScript modes.
#
# We then import the code from `equistore-operations` into a custom module
# `equistore.torch.operations`. When code in this module tries to `from ._classes import
# ...`, the import is resolved to the values defined in the step above. This allows to
# have the right type annotation on the functions.
#
# Overall, the same code is used to define two versions of each function: one will be
# used in `equistore`, and one in `equistore.torch`.


# Step 1: create te `_classes` module as an empty module
spec = importlib.util.spec_from_loader(
    "equistore.torch.operations._classes",
    loader=None,
)
module = importlib.util.module_from_spec(spec)
# This module only exposes a handful of things, defined here. Any changes here MUST also
# be made to the `equistore/operations/_classes.py` file, which is used in non
# TorchScript mode.
module.__dict__["Labels"] = Labels
module.__dict__["TensorBlock"] = TensorBlock
module.__dict__["TensorMap"] = TensorMap
module.__dict__["torch_jit_is_scripting"] = torch.jit.is_scripting


def check_isinstance(obj, ty):
    if isinstance(ty, torch.ScriptClass):
        # This branch is taken when `ty` is a custom class (TensorMap, …). since `ty` is
        # an instance of `torch.ScriptClass` and not a class itself, there is no way to
        # check if obj is an "instance" of this class, so we always return True and hope
        # for the best. Most errors should be caught by the TorchScript compiler anyway.
        return True
    else:
        assert isinstance(ty, type)
        return isinstance(obj, ty)


module.__dict__["check_isinstance"] = check_isinstance

# register the module in sys.modules, so future import find it directly
sys.modules[spec.name] = module


# Step 2: create a module named `equistore.torch.operations` (like the one that will be
# created by importing the current file), but using code from `equistore.operations`
spec = importlib.util.spec_from_file_location(
    # create a module with this name
    "equistore.torch.operations",
    # using the code from there
    equistore.operations.__file__,
)

module = importlib.util.module_from_spec(spec)
# override `equistore.torch.operations` (the module associated with the current file)
# with the newly created module
sys.modules[spec.name] = module
spec.loader.exec_module(module)

# Step 3: override the objects/functions/classes exposed by this module with the data
# from the module we just created.
globals().update(module.__dict__)
