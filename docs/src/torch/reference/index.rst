.. _python-api-torch:

API reference
=============

.. note::

  This is the documentation for ``metatensor-torch`` version
  |metatensor-torch-version|. For other versions, look in the following pages:

  .. version-list::
    :tag-prefix: metatensor-torch-v
    :url-suffix: torch/reference/index.html

    .. version:: 0.8.4
    .. version:: 0.8.3
    .. version:: 0.8.2
    .. version:: 0.8.1
    .. version:: 0.8.0
    .. version:: 0.7.6
    .. version:: 0.7.5
    .. version:: 0.7.4
    .. version:: 0.7.3
    .. version:: 0.7.2
    .. version:: 0.7.1
    .. version:: 0.7.0
    .. version:: 0.6.3
    .. version:: 0.6.1
    .. version:: 0.6.0
    .. version:: 0.5.5
    .. version:: 0.5.4
    .. version:: 0.5.3
    .. version:: 0.5.2
    .. version:: 0.5.1
    .. version:: 0.5.0
    .. version:: 0.4.0
    .. version:: 0.3.0
    .. version:: 0.2.1

    .. version:: 0.2.0
      :url-suffix: reference/torch/index.html

    .. version:: 0.1.0
      :url-suffix: reference/torch/index.html

.. py:currentmodule:: metatensor.torch

The classes and functions in the TorchScript API are kept as close as possible
to the classes and functions of the pure Python API, with the explicit goal that
changing from

.. code-block:: python

    import metatensor as mts
    from metatensor import TensorMap, TensorBlock, Labels

to

.. code-block:: python

    import metatensor.torch as mts
    from metatensor.torch import TensorMap, TensorBlock, Labels

should be 80% of the work required to make a model developed in Python with
:py:mod:`metatensor.operations` compatible with TorchScript. In particular, all
the :ref:`operations <python-api-operations>` are also available in the
``metatensor.torch`` module under the same name. All the functions have the same
behavior, but the versions in ``metatensor.torch`` are annotated with the types
from ``metatensor.torch``, and compatible with TorchScript compilation. For
example :py:func:`metatensor.add()` is available as ``metatensor.torch.add()``.

The :ref:`learn <python-api-learn>` module is also re-exported inside
``metatensor.torch.learn``, with the same functionalities as
``metatensor.learn``.

The documentation for the usual core classes of metatensor can be found in the
following pages:

.. toctree::
    :maxdepth: 1

    tensor
    block
    labels
    serialization


.. this is linked directly from torch/index.rst
.. toctree::
    :maxdepth: 1
    :hidden:

    cxx/index
