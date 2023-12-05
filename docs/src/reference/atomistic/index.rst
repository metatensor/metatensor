.. _atomistic-models:

Atomistic machine learning models
=================================

The atomistic machine learning models capabilities of metatensor are part of the
:ref:`TorchScript API <torch-api-reference>`, and are installed when
:ref:`installing the TorchScript API <install-torch-script>`.

These capabilities can be used from either Python or C++. The Python API is
documented below, and is part of the ``metatensor.torch.atomistic`` module.

.. toctree::
    :maxdepth: 1

    systems
    models
    outputs


C++ interface for atomistic machine learning
--------------------------------------------

The C++ API can be accessed by including ``<metatensor/torch/atomistic.hpp>``.

.. toctree::
    :maxdepth: 1

    cxx/index
