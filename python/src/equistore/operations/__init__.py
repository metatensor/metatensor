"""General operations on :py:class:`TensorMap`"""


from .dot import dot  # noqa
from .lstsq import lstsq  # noqa
from .remove_gradients import remove_gradients  # noqa
from .solve import solve  # noqa


__all__ = ["dot", "lstsq", "remove_gradients", "solve"]
