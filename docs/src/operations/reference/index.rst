.. _python-api-operations:

API reference
=============

.. note::

  This is the documentation for ``metatensor-operations`` version
  |metatensor-operations-version|. For other versions, look in the following
  pages:

  .. version-list::
    :tag-prefix: metatensor-operations-v
    :url-suffix: operations/reference/index.html

    .. version:: 0.3.3
    .. version:: 0.3.2
    .. version:: 0.3.1
    .. version:: 0.3.0
    .. version:: 0.2.4
    .. version:: 0.2.3
    .. version:: 0.2.2
    .. version:: 0.2.1
    .. version:: 0.2.0

    .. version:: 0.1.0
      :url-suffix: reference/operations/index.html

All operations are automatically re-exported from
``metatensor.operations.<xxx>`` as ``metatensor.<xxx>`` when using the Python
backend; and as ``metatensor.torch.<xxx>``` when using the :ref:`TorchScript
backend <operations-and-torch>`.

.. toctree::
    :maxdepth: 2

    creation/index
    linear_algebra/index
    logic/index
    manipulation/index
    math/index
    set/index
    checks
