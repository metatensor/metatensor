"""
.. _core-tutorial-dlpack-interop:

Working with arbitrary types and devices
========================================

Metatensor uses the `DLPack <https://github.com/dmlc/dlpack>`_ standard for
data interchange. This means that data stored inside a ``TensorMap`` is not
restricted to 64-bit floating point numbers on CPU -- it can be **any numeric
type** on **any device**, and can cross language boundaries without copies.

This tutorial shows how this works in practice: storing integer, float16, and
complex data; moving between numpy and torch; and round-tripping through
metatensor's Rust serialization layer without losing type information.

.. py:currentmodule:: metatensor
"""

# %%
#
# Let's start with the necessary imports.

import os
import pathlib
import tempfile

import numpy as np

import metatensor as mts
from metatensor import Labels, TensorBlock, TensorMap


# %%
#
# Storing any numeric type
# ------------------------
#
# Metatensor can store **any numeric type** inside a ``TensorBlock`` — integers,
# half-precision floats, booleans, or complex numbers — and preserves the type
# faithfully through serialization and cross-language boundaries.

int32_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

block_i32 = TensorBlock(
    values=int32_data,
    samples=Labels(["sample"], np.array([[0], [1]], dtype=np.int32)),
    components=[],
    properties=Labels(["property"], np.array([[0], [1], [2]], dtype=np.int32)),
)

print("int32 block dtype:", block_i32.values.dtype)  # int32
print("int32 block values:\n", block_i32.values)

# %%
#
# The same works for float16 (half precision), which is common in ML inference:

f16_data = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float16)

block_f16 = TensorBlock(
    values=f16_data,
    samples=Labels.range("sample", 2),
    components=[],
    properties=Labels.range("property", 2),
)

print("float16 block dtype:", block_f16.values.dtype)  # float16
print("float16 block values:\n", block_f16.values)

# %%
#
# And for complex numbers, used in quantum chemistry and signal processing:

complex_data = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex128)

block_complex = TensorBlock(
    values=complex_data,
    samples=Labels.range("sample", 1),
    components=[],
    properties=Labels.range("property", 2),
)

print("complex128 block dtype:", block_complex.values.dtype)  # complex128
print("complex128 block values:\n", block_complex.values)


# %%
#
# Serialization preserves types
# -----------------------------
#
# When you save a ``TensorMap`` to disk, the dtype of every block's values is
# preserved exactly. The path through the code is:
#
# **Python array** → (DLPack) → **C API** → **Rust** → **.npy** inside a .mts file
#
# and on loading, the reverse. At every boundary crossing, DLPack carries the
# type information, so nothing is lost.

keys = Labels(["type"], np.array([[0]], dtype=np.int32))
tensor_i32 = TensorMap(keys, [block_i32.copy()])

# use_numpy=False means: Python -> DLPack -> C API -> Rust -> .npy serialization
mts.save("int32_tensor.mts", tensor_i32, use_numpy=False)

# use_numpy=True means: .npy deserialization -> numpy (preserves dtype natively)
loaded = mts.load("int32_tensor.mts", use_numpy=True)

print("Saved dtype: ", int32_data.dtype)
print("Loaded dtype:", loaded.block(0).values.dtype)
print("Values match:", np.array_equal(loaded.block(0).values, int32_data))


# %%
#
# This works for every supported type. Here is a quick round-trip test for
# several dtypes:

dtypes = [
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int32,
    np.int64,
    np.uint8,
    np.bool_,
    np.complex64,
    np.complex128,
]

with tempfile.TemporaryDirectory() as tmp:
    for dtype in dtypes:
        data = np.arange(6).reshape(2, 3).astype(dtype)
        block = TensorBlock(
            values=data,
            samples=Labels.range("s", 2),
            components=[],
            properties=Labels.range("p", 3),
        )
        tensor = TensorMap(keys, [block])

        path = os.path.join(tmp, f"{dtype.__name__}.mts")
        mts.save(path, tensor, use_numpy=False)
        loaded = mts.load(path, use_numpy=True)

        assert loaded.block(0).values.dtype == dtype
        print(f"  {dtype.__name__:>12s}: round-trip OK")


# %%
#
# Numpy and Torch interoperability
# ---------------------------------
#
# If PyTorch is available, you can also create blocks from torch tensors. The
# DLPack layer handles the conversion transparently -- the data is shared, not
# copied.

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    # Create a torch tensor with float32 dtype
    torch_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    block_torch = TensorBlock(
        values=torch_data,
        samples=Labels.range("sample", 2),
        components=[],
        properties=Labels.range("property", 2),
    )

    print("Torch block dtype:", block_torch.values.dtype)
    print("Torch block values:\n", block_torch.values)
    print("Is a torch.Tensor:", isinstance(block_torch.values, torch.Tensor))

# %%
#
# You can convert between numpy and torch backends using ``.to()``:

if HAS_TORCH:
    # torch float32 -> numpy float32
    as_numpy = block_torch.to(arrays="numpy")
    print("Converted to numpy:", type(as_numpy.values).__name__, as_numpy.values.dtype)

    # numpy float16 -> torch float16
    as_torch = block_f16.to(arrays="torch")
    print("Converted to torch:", type(as_torch.values).__name__, as_torch.values.dtype)


# %%
#
# GPU tensors (if available)
# --------------------------
#
# With PyTorch, metatensor can hold data on GPU. The DLPack metadata carries the
# device information, so metatensor knows where the data lives and can handle it
# correctly.

if HAS_TORCH and torch.cuda.is_available():
    gpu_data = torch.randn(3, 4, dtype=torch.float32, device="cuda")

    block_gpu = TensorBlock(
        values=gpu_data,
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels.range("property", 4),
    )

    print("GPU block device:", block_gpu.device)
    print("GPU block dtype: ", block_gpu.dtype)

    # Move to CPU and convert to numpy in one step
    block_cpu = block_gpu.to("cpu").to(arrays="numpy")
    print("After .to('cpu').to(arrays='numpy'):", type(block_cpu.values).__name__)

elif HAS_TORCH:
    print("(CUDA not available, skipping GPU example)")
else:
    print("(torch not available, skipping GPU example)")


# %%
#
# How DLPack enables this
# -----------------------
#
# Under the hood, every data array in metatensor implements a single interface:
# ``as_dlpack``. This is part of the `DLPack standard
# <https://github.com/dmlc/dlpack>`_, which encodes:
#
# - A **pointer** to the raw data buffer
# - The **dtype** (integer, float, complex, bool) and bit width (8, 16, 32, 64)
# - The **device** type and ID (CPU, CUDA, ROCm, Metal, etc.)
# - The **shape** and **strides** of the array
#
# This metadata travels across every FFI boundary -- Python to C, C to Rust,
# Rust to C++ -- without losing information. The array data itself is **never
# copied** during these transitions; only the thin DLPack descriptor is passed.
#
# This is what makes the following pipeline work for *any* type and device:
#
# .. code-block:: text
#
#     Python (numpy/torch)
#       │  ── DLPack ──►  C API (mts_array_t.as_dlpack)
#       │                    │  ── DLPack ──►  Rust (serialization / operations)
#       │                    │                    │  ── DLPack ──►  .npy files
#       │                    │  ◄── DLPack ──  Rust
#       │  ◄── DLPack ──  C API
#     Python
#
# Each arrow is a zero-copy handoff. The type and device information is preserved
# at every step because DLPack's ``DLDataType`` and ``DLDevice`` structs are
# part of the tensor descriptor.
#

# %%
#
# Cleanup


pathlib.Path("int32_tensor.mts").unlink(missing_ok=True)
