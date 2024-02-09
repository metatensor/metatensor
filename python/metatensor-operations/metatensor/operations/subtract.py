from typing import Union

from ._backend import (
    TensorMap,
    check_isinstance,
    torch_jit_is_scripting,
    torch_jit_script,
)
from .add import add
from .multiply import multiply


@torch_jit_script
def subtract(A: TensorMap, B: Union[float, int, TensorMap]) -> TensorMap:
    r"""Return a new :class:`TensorMap` with the values being the subtract
    of ``A`` and ``B``.

    If ``B`` is a :py:class:`TensorMap` it has to have the same metadata as ``A``.

    If gradients are present in ``A``:

    *  ``B`` is a scalar:

       .. math::
            \nabla(A - B) = \nabla A

    * ``B`` is a :py:class:`TensorMap` with the same metadata of ``A``:

       .. math::
            \nabla(A - B) = \nabla A - \nabla B

    :param A: First :py:class:`TensorMap` for the subtraction.
    :param B: Second instance for the subtraction. Parameter can be a scalar or a
              :py:class:`TensorMap`. In the latter case ``B`` must have the same
              metadata of ``A``.

    :return: New :py:class:`TensorMap` with the same metadata as ``A``.
    """
    if not torch_jit_is_scripting():
        if not check_isinstance(A, TensorMap):
            raise TypeError(f"`A` must be a metatensor TensorMap, not {type(A)}")

    if torch_jit_is_scripting():
        is_tensor_map = isinstance(B, TensorMap)
    else:
        is_tensor_map = check_isinstance(B, TensorMap)

    if isinstance(B, (float, int)):
        B = -float(B)
    elif is_tensor_map:
        B = multiply(B, -1)
    else:
        if torch_jit_is_scripting():
            extra = ""
        else:
            extra = f", not {type(B)}"

        raise TypeError("`B` must be a metatensor TensorMap or a scalar value" + extra)

    return add(A=A, B=B)
