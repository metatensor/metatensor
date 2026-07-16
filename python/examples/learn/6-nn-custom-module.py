"""
.. _learn-tutorial-nn-custom-module:

Custom modules with metatensor data
===================================

.. py:currentmodule:: metatensor.torch.learn.nn

This tutorial explains how to store metatensor data (``Labels``, ``TensorBlock``,
``TensorMap``) inside your own custom modules, so that it is correctly handled by
``.to()``, ``state_dict()``, and TorchScript.

.. note::

    This tutorial builds on the concepts introduced in :ref:`convenience modules
    <learn-tutorial-nn-modules-basic>` and :ref:`custom architectures with ModuleMap
    <learn-tutorial-nn-using-modulemap>`.
"""

import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.learn import nn


torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


# the dtype of TensorMap/TensorBlock is returned as an int, we can use this function to
# convert torch dtype to the same int for comparison
@torch.jit.script
def dtype_to_int(dtype: torch.dtype):
    return dtype


# %%
#
# Why ``nn.Module``?
# -------------------
#
# When a module holds metatensor data as plain attributes on a standard
# ``torch.nn.Module``, calling ``.to()`` or ``.cuda()`` will **not** move the metatensor
# data, and ``state_dict()`` will **not** include it. The
# :py:class:`metatensor.torch.learn.nn.Module` class solves this by intercepting
# ``.to()`` and ``state_dict()`` / ``load_state_dict()``.
#
# To make this work, metatensor data must be registered explicitly via
# :py:meth:`~metatensor.torch.learn.nn.Module.register_buffer`, mirroring how PyTorch
# handles regular tensor buffers.


# %%
#
# A simple custom module
# -----------------------
#
# Let's build a linear layer that stores its output properties as a ``Labels`` buffer.
# This is a common pattern: the labels describe the output dimension and should follow
# the module when it is moved across devices or saved to a checkpoint. This is also what
# the :py:class:`~metatensor.torch.learn.nn.Linear` convenience module does.


class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.register_buffer(
            "properties",
            Labels("out_features", torch.arange(out_features).reshape(-1, 1)),
        )
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, tensor: TensorMap) -> TensorMap:
        blocks = []
        for block in tensor:
            new_values = self.linear(block.values)
            new_block = TensorBlock(
                new_values,
                block.samples,
                block.components,
                self.properties,
            )
            blocks.append(new_block)
        return TensorMap(tensor.keys, blocks)


# Create a dummy input TensorMap
feature_tensor = torch.randn(100, 128)
feature_tensormap = TensorMap(
    keys=Labels.single(),
    blocks=[mts.block_from_array(feature_tensor)],
)

model = CustomLinear(in_features=128, out_features=4)
output = model(feature_tensormap)
print("output block shape:", output.block(0).values.shape)
print("properties:", output.block(0).properties)


# %%
#
# Moving data with ``.to()``
# --------------------------
#
# Because ``properties`` is a registered buffer, calling ``.to()`` moves it
# along with the rest of the module (labels, values, etc.):

model = model.to(dtype=torch.float32)
# the dtype does not change, Labels are always int32
assert model.properties.values.dtype == torch.int32
assert model.linear.weight.dtype == torch.float32

if torch.cuda.is_available():
    model = model.to(device="cuda")
    # the device is properly updated for both the Labels and the torch tensor
    assert model.properties.device.type == "cuda"
    assert model.linear.weight.device.type == "cuda"

# %%
#
# Saving and loading with ``state_dict``
# ---------------------------------------
#
# Metatensor buffers are included in ``state_dict()`` and restored by
# ``load_state_dict()``.

state_dict = model.state_dict()
# The metatensor buffer appears under the "_extra_state" key
print("metatensor extra state:", list(state_dict["_extra_state"].keys()))

# Create a fresh model and load the state dict
model2 = CustomLinear(in_features=128, out_features=4)
model2.load_state_dict(state_dict)
assert model2.properties == model.properties


# %%
#
# Storing ``TensorBlock`` and ``TensorMap``
# ------------------------------------------
#
# The same mechanism works for ``TensorBlock`` and ``TensorMap`` values, as
# well as for (arbitrarily nested) ``dict``, ``list``, and ``tuple``
# containers of metatensor data.


class CustomModule(nn.Module):
    def __init__(self, block: TensorBlock, tensor: TensorMap):
        super().__init__()
        # Plain metatensor values
        self.register_buffer("block", block)
        self.register_buffer("tensor", tensor)
        # Nested containers are also supported
        self.register_buffer("nested", {"block": block, "tensor": tensor})

    def forward(self, x: int) -> TensorBlock:
        if x == 0:
            return self.block
        else:
            return self.nested["block"]


def _make_block():
    return TensorBlock(
        values=torch.randn(3, 4),
        samples=Labels(["s"], torch.arange(3).reshape(-1, 1)),
        components=[],
        properties=Labels(["p"], torch.arange(4).reshape(-1, 1)),
    )


block = _make_block()
tensor = TensorMap(Labels.single(), [_make_block()])

module = CustomModule(block, tensor)
module = module.to(dtype=torch.float32)
# All metatensor data has been moved to the new dtype
assert module.block.dtype == dtype_to_int(torch.float32)
assert module.tensor.dtype == dtype_to_int(torch.float32)
assert module.nested["block"].dtype == dtype_to_int(torch.float32)

if torch.cuda.is_available():
    module = module.to(device="cuda")
    # the device is properly updated for both the Labels and the torch tensor
    assert module.block.device.type == "cuda"
    assert module.tensor.device.type == "cuda"
    assert module.nested["block"].device.type == "cuda"


# %%
#
# The ``nn.Buffer`` shorthand
# ---------------------------
#
# Instead of calling ``register_buffer`` explicitly, you can assign an
# :py:class:`~metatensor.torch.learn.nn.Buffer` wrapper directly as an attribute. This
# is equivalent and can be more concise:


class ShorthandModule(nn.Module):
    def __init__(self, labels: Labels):
        super().__init__()

        # same as self.register_buffer("labels", labels)
        self.labels = nn.Buffer(labels)

    def forward(self) -> Labels:
        return self.labels


shorthand = ShorthandModule(Labels("p", torch.arange(4).reshape(-1, 1)))
assert shorthand.labels.names == ["p"]


# %%
#
# Non-persistent buffers
# ----------------------
#
# Just like PyTorch, ``register_buffer`` accepts a ``persistent`` argument.
# Non-persistent buffers are excluded from ``state_dict`` but are still moved by
# ``.to()``. This is useful for cached or computed data that should not be saved in
# checkpoints.


class WithCache(nn.Module):
    def __init__(self, labels: Labels, cache: Labels):
        super().__init__()
        self.register_buffer("labels", labels, persistent=True)
        self.register_buffer("cached", cache, persistent=False)

    def forward(self, x: int) -> Labels:
        if x == 0:
            return self.labels
        return self.cached


cached = WithCache(
    Labels("p", torch.arange(4).reshape(-1, 1)),
    Labels("c", torch.arange(2).reshape(-1, 1)),
)
state_dict = cached.state_dict()
assert "labels" in state_dict["_extra_state"]
assert "cached" not in state_dict["_extra_state"]

# Both buffers are still moved by .to()
cached = cached.to(dtype=torch.float32)
assert cached.labels.device.type == "cpu"
assert cached.cached.device.type == "cpu"


# %%
#
# ``Module`` vs ``ModuleMap``
# ---------------------------
#
# The :py:class:`~metatensor.torch.learn.nn.Module` class is a general-purpose
# base class: it handles metatensor buffers on any module that stores them, and
# the ``forward`` method can take and return anything, including
# :py:class:`~metatensor.torch.TensorMap` objects.
#
# The :py:class:`~metatensor.torch.learn.nn.ModuleMap` class is more specialized:
# it wraps a collection of standard ``torch.nn.Module`` objects (one per key in
# a ``TensorMap``) and applies the corresponding module to each
# :py:class:`~metatensor.torch.TensorBlock` independently. In other words,
# ``ModuleMap`` operates on the **blocks** of a ``TensorMap``, while
# ``Module`` operates on the **module itself**.
#
# The convenience layers seen in :ref:`nn modules basic
# <learn-tutorial-nn-modules-basic>` (``Linear``, ``ReLU``, ``Sequential``,
# etc.) and the custom architectures in :ref:`custom architectures with
# ModuleMap <learn-tutorial-nn-using-modulemap>` all use ``ModuleMap`` under
# the hood, which itself inherits from ``Module``. This means that if your
# custom module operates on individual blocks and you want to reuse the
# ``TensorMap`` plumbing, you should reach for ``ModuleMap``. If you need full
# control over the ``forward`` logic or store metatensor data that is not
# block-wise, use ``Module`` directly.


# %%
#
# Conclusion
# ----------
#
# Use :py:class:`~metatensor.torch.learn.nn.Module` with
# :py:meth:`~metatensor.torch.learn.nn.Module.register_buffer` whenever your module
# stores ``Labels``, ``TensorBlock``, or ``TensorMap`` data. This ensures the data is
# moved by ``.to()`` and included in ``state_dict()`` -- all with a single explicit call
# to ``register_buffer``, matching standard PyTorch conventions.
