"""
Registration of :py:class:`TensorBlock` and :py:class:`TensorMap` as JAX pytrees, so
they can be given to and returned by :py:func:`jax.jit`, :py:func:`jax.grad`,
:py:func:`jax.vmap`, ... The arrays are the leaves, and the metadata is static data.

JAX is an optional dependency, and this module is only used once JAX was imported.
"""

_REGISTERED = False


def register_pytrees():
    """
    Register :py:class:`TensorBlock` and :py:class:`TensorMap` as JAX pytrees. This is
    called when metatensor sees a JAX array for the first time, or when metatensor is
    imported after JAX, and does nothing if the classes are already registered.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    import jax

    from ._block import TensorBlock
    from ._tensor import TensorMap

    jax.tree_util.register_pytree_node(TensorBlock, _flatten_block, _unflatten_block)
    jax.tree_util.register_pytree_node(TensorMap, _flatten_tensor, _unflatten_tensor)


class _StaticMetadata:
    """
    Metadata of a block or tensor map, stored as the static data of the pytree. JAX
    requires this data to be hashable, which :py:class:`Labels` are not: the hash only
    uses the names and size of the labels, and the equality compares them fully.
    """

    __slots__ = ["labels", "extra"]

    def __init__(self, labels, extra=()):
        self.labels = tuple(labels)
        self.extra = tuple(extra)

    def __hash__(self):
        shapes = tuple((tuple(labels.names), len(labels)) for labels in self.labels)
        return hash((shapes, self.extra))

    def __eq__(self, other):
        return (
            isinstance(other, _StaticMetadata)
            and self.extra == other.extra
            and len(self.labels) == len(other.labels)
            and all(a == b for a, b in zip(self.labels, other.labels, strict=True))
        )


def _is_array(value):
    from ._data._array import _is_jax_array, _is_numpy_array, _is_torch_array

    return _is_numpy_array(value) or _is_torch_array(value) or _is_jax_array(value)


def _placeholder(cls, children, metadata):
    """
    JAX can rebuild pytrees with leaves that are not arrays, e.g. shapes in
    :py:func:`jax.eval_shape`, or anything returned by :py:func:`jax.tree.map`.
    Metatensor can not store these, so we create an instance of ``cls`` without any
    data instead, which only keeps the children and metadata to be flattened again.
    """
    placeholder = cls.__new__(cls)
    placeholder._ptr = None
    placeholder._pytree = (children, metadata)
    return placeholder


def _is_placeholder(value):
    return getattr(value, "_pytree", None) is not None


def _flatten_block(block):
    if _is_placeholder(block):
        return block._pytree

    # the gradients are blocks themselves, and are flattened recursively
    parameters = block.gradients_list()
    children = [block.values] + [block.gradient(p) for p in parameters]
    metadata = _StaticMetadata(
        [block.samples, block.properties, *block.components],
        extra=parameters,
    )
    return children, metadata


def _unflatten_block(metadata, children):
    from ._block import TensorBlock

    samples, properties, *components = metadata.labels
    values, *gradients = children

    shape = (len(samples), *(len(c) for c in components), len(properties))
    if (
        not _is_array(values)
        or tuple(values.shape) != shape
        or any(_is_placeholder(gradient) for gradient in gradients)
    ):
        return _placeholder(TensorBlock, children, metadata)

    block = TensorBlock(values, samples, components, properties)
    for parameter, gradient in zip(metadata.extra, gradients, strict=True):
        block.add_gradient(parameter, gradient)
    return block


def _flatten_tensor(tensor):
    if _is_placeholder(tensor):
        return tensor._pytree

    return tensor.blocks(), _StaticMetadata([tensor.keys])


def _unflatten_tensor(metadata, blocks):
    from ._tensor import TensorMap

    if any(_is_placeholder(block) for block in blocks):
        return _placeholder(TensorMap, blocks, metadata)

    (keys,) = metadata.labels
    return TensorMap(keys, list(blocks))
