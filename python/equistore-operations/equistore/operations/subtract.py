from typing import Union

from ._classes import TensorMap
from .add import add
from .multiply import multiply


def subtract(A: TensorMap, B: Union[float, TensorMap]) -> TensorMap:
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
    if isinstance(B, TensorMap):
        B = multiply(B, -1)
    elif isinstance(B, (float, int)):
        B = -float(B)
    else:
        raise TypeError("B should be a TensorMap or a scalar value")

    tensor_result = add(A=A, B=B)

    return tensor_result
