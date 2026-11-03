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


def _flatten_block(block):
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

    block = TensorBlock(values, samples, components, properties)
    for parameter, gradient in zip(metadata.extra, gradients, strict=True):
        block.add_gradient(parameter, gradient)
    return block


def _flatten_tensor(tensor):
    return tensor.blocks(), _StaticMetadata([tensor.keys])


def _unflatten_tensor(metadata, blocks):
    from ._tensor import TensorMap

    (keys,) = metadata.labels
    return TensorMap(keys, list(blocks))
