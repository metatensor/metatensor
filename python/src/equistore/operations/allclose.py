import numpy as np

from equistore import TensorBlock, TensorMap

from . import _dispatch
from .equal_metadata import _check_blocks, _check_maps, _check_same_gradients


def allclose(
    tensor1: TensorMap,
    tensor2: TensorMap,
    rtol=1e-13,
    atol=1e-12,
    equal_nan=False,
) -> bool:
    """Compare two :py:class:`TensorMap`.

    This function returns ``True`` if the two tensors have the same keys
    (potentially in different order) and all the :py:class:`TensorBlock` have
    the same (and in the same order) samples, components, properties,
    and their values matrices pass the numpy-like ``allclose`` test with the provided
    ``rtol``, and ``atol``.

    The :py:class:`TensorMap` contains gradient data, then this function only
    returns ``True`` if all the gradients also have the same samples,
    components, properties and their data matrices pass the numpy-like
    ``allclose`` test with the provided ``rtol``, and ``atol``.

    In practice this function calls :py:func:`allclose_raise`, returning
    ``True`` if no exception is rased, `False` otherwise.

    Here are examples using this function:

    >>> from equistore import Labels
    >>> # Create simple block
    >>> block1 = TensorBlock(
    ...    values=np.array([
    ...         [1, 2, 4],
    ...         [3, 5, 6],
    ...     ]),
    ...    samples=Labels(
    ...         ["structure", "center"],
    ...         np.array([
    ...             [0, 0],
    ...             [0, 1],
    ...         ]),
    ...     ),
    ...    components=[],
    ...    properties=Labels(
    ...        ["properties"], np.array([[0], [1], [2]])
    ...     ),
    ... )
    >>> # Create a second block that is equivalent to block1
    >>> block2 = TensorBlock(
    ...     values=np.array([
    ...          [1, 2, 4],
    ...          [3, 5, 6],
    ...      ]),
    ...     samples=Labels(
    ...          ["structure", "center"],
    ...          np.array([
    ...              [0, 0],
    ...              [0, 1],
    ...          ]),
    ...      ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["properties"], np.array([[0], [1], [2]])
    ...      ),
    ... )
    ...
    >>> # Create tensors from blocks, using keys with different names
    >>> keys1 = Labels(names=["key1"], values=np.array([[0]]))
    >>> keys2 = Labels(names=["key2"], values=np.array([[0]]))
    ...
    >>> tensor1 = TensorMap(keys1, [block1])
    >>> tensor2 = TensorMap(keys2, [block2])
    ...
    >>> # Call allclose, which should fail as the blocks have different keys
    >>> # associated with them
    >>> allclose(tensor1, tensor2)
    False
    >>> # Create a third tensor, which differs from tensor1 only by 1e-5 in a
    >>> # single block value
    >>> block3 = TensorBlock(
    ...     values=np.array([
    ...          [1+1e-5, 2, 4],
    ...          [3, 5, 6],
    ...      ]),
    ...     samples=Labels(
    ...          ["structure", "center"],
    ...          np.array([
    ...              [0, 0],
    ...              [0, 1],
    ...          ]),
    ...      ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["properties"], np.array([[0], [1], [2]])
    ...      ),
    ... )
    ...
    >>> #Create tensors from blocks, using key with same name as block1
    >>> keys3 = Labels(names=["key1"], values=np.array([[0]]))
    ...
    >>> tensor3 = TensorMap(keys3, [block3])
    >>> # Call allclose, which should return False because the default rtol
    >>> # is 1e-13, and the difference in the first value between the blocks
    >>> # of the two tensors is 1e-5
    >>> allclose(tensor1, tensor3)
    False
    >>> # Calling allclose again with the optional argument rtol=1e-5 should
    >>> # return True, as the difference in the first value between the blocks
    >>> # of the two tensors is within the tolerance limit
    >>> allclose(tensor1, tensor3, rtol=1e-5)
    True

    :param tensor1: first :py:class:`TensorMap`.
    :param tensor2: second :py:class:`TensorMap`.
    :param rtol: relative tolerance for ``allclose``. Default: 1e-13.
    :param atol: absolute tolerance for ``allclose``. Defaults: 1e-12.
    :param equal_nan: should two ``NaN`` be considered equal? Defaults: False.
    """
    try:
        allclose_raise(
            tensor1=tensor1,
            tensor2=tensor2,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
        return True
    except ValueError:
        return False


def allclose_raise(
    tensor1: TensorMap,
    tensor2: TensorMap,
    rtol=1e-13,
    atol=1e-12,
    equal_nan=False,
):
    """
    Compare two :py:class:`TensorMap`, raising a ``ValueError`` if they are not
    the same.

    The message associated with the ``ValueError`` will contain more information
    on where the two :py:class:`TensorMap` differ. See :py:func:`allclose` for
    more information on which :py:class:`TensorMap` are considered equal.

    Here are examples using this function:

    >>> from equistore import Labels
    >>> # Create simple block, with one np.nan value
    >>> block1 = TensorBlock(
    ...     values=np.array([
    ...          [1, 2, 4],
    ...          [3, 5, np.nan],
    ...      ]),
    ...     samples=Labels(
    ...          ["structure", "center"],
    ...          np.array([
    ...              [0, 0],
    ...              [0, 1],
    ...          ]),
    ...      ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["properties"], np.array([[0], [1], [2]])
    ...      ),
    ... )
    ...
    >>> # Create a second block that differs from block1 by 1e-5 in its
    >>> # first value
    >>> block2 = TensorBlock(
    ...     values=np.array([
    ...          [1+1e-5, 2, 4],
    ...          [3, 5, np.nan],
    ...      ]),
    ...     samples=Labels(
    ...          ["structure", "center"],
    ...          np.array([
    ...              [0, 0],
    ...              [0, 1],
    ...          ]),
    ...      ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["properties"], np.array([[0], [1], [2]])
    ...      ),
    ... )
    ...
    >>> #Create tensors from blocks, using same keys
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    ...
    >>> tensor1 = TensorMap(keys, [block1])
    >>> tensor2 = TensorMap(keys, [block2])
    ...
    >>> # Call allclose_raise, which should return a ValueError because:
    >>> # 1. The two NaNs are not considered equal,
    >>> # 2. The difference between the first value in the blocks
    >>> # is greater than the default rtol of 1e-13
    >>> # If this is executed yourself, you will see a nested exception
    >>> # ValueError which explains that the values of the two blocks
    >>> # are not allclose
    >>> allclose_raise(tensor1, tensor2)
    Traceback (most recent call last):
        ...
    ValueError: The TensorBlocks with key = (0,) are different
    >>> # Call allclose_raise again, but use equal_nan=True and rtol=1e-5
    >>> # This passes, as the two NaNs are now considered equal, and the
    >>> # difference between the first values of the blocks of the two tensors
    >>> # is within the rtol limit of 1e-5
    >>> allclose_raise(tensor1, tensor2, equal_nan=True, rtol=1e-5)

    :param tensor1: first :py:class:`TensorMap`.
    :param tensor2: second :py:class:`TensorMap`.
    :param rtol: relative tolerance for ``allclose``. Default: 1e-13.
    :param atol: absolute tolerance for ``allclose``. Defaults: 1e-12.
    :param equal_nan: should two ``NaN`` be considered equal? Defaults: False.
    """
    _check_maps(tensor1, tensor2, "allclose")

    for key, block1 in tensor1:
        try:
            allclose_block_raise(
                block1,
                tensor2.block(key),
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
        except ValueError as e:
            raise ValueError(f"The TensorBlocks with key = {key} are different") from e


def allclose_block(
    block1: TensorBlock,
    block2: TensorBlock,
    rtol=1e-13,
    atol=1e-12,
    equal_nan=False,
) -> bool:
    """
    Compare two :py:class:`TensorBlock`.

    This function returns ``True`` if the two :py:class:`TensorBlock` have the
    same samples, components, properties and their values matrices must pass the
    numpy-like ``allclose`` test with the provided ``rtol``, and ``atol``.

    If the :py:class:`TensorBlock` contains gradients, then the gradient must
    also have same (and in the same order) samples, components, properties
    and their data matrices must pass the numpy-like ``allclose`` test with the
    provided ``rtol``, and ``atol``.

    In practice this function calls :py:func:`allclose_block_raise`, returning
    ``True`` if no exception is rased, `False` otherwise.

    Here are some exmaples using this function:

    >>> from equistore import Labels
    >>> # Create simple block
    >>> block1 = TensorBlock(
    ...     values=np.array([
    ...          [1, 2, 4],
    ...          [3, 5, 6],
    ...      ]),
    ...     samples=Labels(
    ...          ["structure", "center"],
    ...          np.array([
    ...              [0, 0],
    ...              [0, 1],
    ...          ]),
    ...      ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["property_1"], np.array([[0], [1], [2]])
    ...      ),
    ... )
    ...
    >>> # Recreate block1, but change first value in the block from 1 to 1.00001
    >>> block2 = TensorBlock(
    ...     values=np.array([
    ...          [1+1e-5, 2, 4],
    ...          [3, 5, 6],
    ...      ]),
    ...     samples=Labels(
    ...          ["structure", "center"],
    ...          np.array([
    ...              [0, 0],
    ...              [0, 1],
    ...          ]),
    ...      ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["property_1"], np.array([[0], [1], [2]])
    ...      ),
    ... )
    ...
    >>> # Call allclose_block, which should return False because the default
    >>> # rtol is 1e-13, and the difference in the first value between the
    >>> # two blocks is 1e-5
    >>> allclose_block(block1, block2)
    False
    >>> # Calling allclose_block with the optional argument rtol=1e-5 should
    >>> # return True, as the difference in the first value between the two
    >>> # blocks is within the tolerance limit
    >>> allclose_block(block1, block2, rtol=1e-5)
    True

    :param block1: first :py:class:`TensorBlock`.
    :param block2: second :py:class:`TensorBlock`.
    :param rtol: relative tolerance for ``allclose``. Default: 1e-13.
    :param atol: absolute tolerance for ``allclose``. Defaults: 1e-12.
    :param equal_nan: should two ``NaN`` be considered equal? Defaults: False.
    """
    try:
        allclose_block_raise(
            block1=block1,
            block2=block2,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
        return True
    except ValueError:
        return False


def allclose_block_raise(
    block1: TensorBlock,
    block2: TensorBlock,
    rtol=1e-13,
    atol=1e-12,
    equal_nan=False,
):
    """
    Compare two :py:class:`TensorBlock`, raising a ``ValueError`` if they are
    not the same.

    The message associated with the ``ValueError`` will contain more information
    on where the two :py:class:`TensorBlock` differ. See
    :py:func:`allclose_block` for more information on which
    :py:class:`TensorBlock` are considered equal.

    Here is an example using this function:

    >>> from equistore import Labels
    >>> #Create simple block
    >>> block1 = TensorBlock(
    ...     values=np.array([
    ...          [1, 2, 4],
    ...          [3, 5, 6],
    ...      ]),
    ...     samples=Labels(
    ...          ["structure", "center"],
    ...          np.array([
    ...              [0, 0],
    ...              [0, 1],
    ...          ]),
    ...      ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["property_1"], np.array([[0], [1], [2]])
    ...      ),
    ... )
    ...
    >>> #Recreate block1, but rename properties label 'property_1' to 'property_2'
    >>> block2 = TensorBlock(
    ...     values=np.array([
    ...          [1, 2, 4],
    ...          [3, 5, 6],
    ...      ]),
    ...     samples=Labels(
    ...          ["structure", "center"],
    ...          np.array([
    ...              [0, 0],
    ...              [0, 1],
    ...          ]),
    ...      ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["property_2"], np.array([[0], [1], [2]])
    ...      ),
    ... )
    ...
    >>> # Call allclose_block_raise, which should raise a ValueError because the
    >>> # properties of the two blocks are not equal
    >>> allclose_block_raise(block1, block2)
    Traceback (most recent call last):
        ...
    ValueError: Inputs to 'allclose' should have the same properties:
    properties names are not the same or not in the same order.

    :param block1: first :py:class:`TensorBlock`.
    :param block2: second :py:class:`TensorBlock`.
    :param rtol: relative tolerance for ``allclose``. Default: 1e-13.
    :param atol: absolute tolerance for ``allclose``. Defaults: 1e-12.
    :param equal_nan: should two ``NaN`` be considered equal? Defaults: False.
    """

    if not np.all(block1.values.shape == block2.values.shape):
        raise ValueError("values shapes are different")

    if not _dispatch.allclose(
        block1.values,
        block2.values,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    ):
        raise ValueError("values are not allclose")
    _check_blocks(
        block1, block2, props=["samples", "properties", "components"], fname="allclose"
    )
    _check_same_gradients(
        block1, block2, props=["samples", "properties", "components"], fname="allclose"
    )

    for parameter, gradient1 in block1.gradients():
        gradient2 = block2.gradient(parameter)

        if not _dispatch.allclose(
            gradient1.data,
            gradient2.data,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        ):
            raise ValueError(f'gradient ("{parameter}") data are not allclose')
