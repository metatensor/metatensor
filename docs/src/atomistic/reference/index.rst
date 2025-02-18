.. _python-api-atomistic-models:

API reference
=============

.. note::

  This is the documentation for the atomistic capabilities of
  ``metatensor-torch`` version |metatensor-torch-version|. For other versions,
  look in the following pages:


  .. version-list::
    :tag-prefix: metatensor-torch-v
    :url-suffix: atomistic/reference/index.html

    .. version:: 0.7.2
    .. version:: 0.7.1
    .. version:: 0.7.0
    .. version:: 0.6.3
    .. version:: 0.6.2
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


The atomistic machine learning models capabilities of metatensor are part of the
:ref:`TorchScript API <torch-api-reference>`, and are installed when
:ref:`installing the TorchScript API <install-torch>`. We provide code for using
atomistic systems as input of a machine learning model with
:py:class:`metatensor.torch.atomistic.System`, and exporting trained models with
:py:class:`metatensor.torch.atomistic.MetatensorAtomisticModel`.

.. toctree::
    :maxdepth: 1

    systems
    models/index
    ase
    serialization


C++ API reference
-----------------

The C++ API can be accessed by including ``<metatensor/torch/atomistic.hpp>``.

.. toctree::
    :maxdepth: 1

    cxx/index
