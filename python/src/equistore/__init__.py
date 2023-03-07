"""
Most users will find the Python interface to ``equistore`` to be the most
convenient to use. This interface is built on top of the C API, and can be
:ref:`installed independently <install-python-lib>`. The functions and classes
provided in ``equistore`` can be grouped into the following:

First the three core objects :py:class:`equistore.TensorMap`,
:py:class:`equistore.TensorBlock`, :py:class:`equistore.Labels` of equistore.

:ref:`Operations <python-api-operations>` include mathematical, logical as well as
utility operations that can applied on the three core objects.

The API also includes more advanced functionalities like
:ref:`IO operations<python-api-io>` or the
:ref:`actual array format <python-api-array>` for storing data.
"""

from .version import __version__  # noqa
from .block import TensorBlock  # noqa
from .labels import Labels  # noqa
from .status import EquistoreError  # noqa
from .tensor import TensorMap  # noqa

from .operations import *  # noqa


__all__ = ["TensorBlock", "Labels", "TensorMap"]
