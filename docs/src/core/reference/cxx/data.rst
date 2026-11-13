Data arrays
===========

.. doxygenclass:: metatensor::DataArrayBase
    :members:

.. doxygenclass:: metatensor::MtsArray
    :members:

--------------------------------------------------------------------------------

.. doxygenclass:: metatensor::SimpleDataArray
    :members: SimpleDataArray, operator=, view, from_mts_array

.. doxygenclass:: metatensor::EmptyDataArray
    :members: EmptyDataArray, operator=


Kokkos arrays
-------------

:cpp:class:`metatensor::KokkosDataArray` stores data in a `Kokkos`_ view, on CPU
or GPU, using the CUDA (NVIDIA GPUs), HIP (AMD GPUs) or SYCL (Intel GPUs) Kokkos
backends. Using the SYCL backend on NVIDIA or AMD GPUs is not supported, please
use the CUDA or HIP backend instead. This class is defined in a separate header,
which is not included by ``<metatensor.hpp>``. Metatensor does not distribute
Kokkos, so you need to make Kokkos (version 4.0 or later) available in the
include path yourself, for example by linking to the ``Kokkos::kokkos`` CMake
target:

.. code-block:: cmake

    find_package(Kokkos REQUIRED)
    find_package(metatensor REQUIRED)

    target_link_libraries(your-target metatensor Kokkos::kokkos)

.. code-block:: cpp

    #include <metatensor.hpp>
    #include <metatensor/kokkos.hpp>

    auto view = Kokkos::View<double**>("values", n_samples, n_properties);
    // ... fill the view

    auto block = metatensor::TensorBlock(
        std::make_unique<metatensor::KokkosDataArray<double>>(view),
        samples, {}, properties
    );

All Kokkos views must be released before calling ``Kokkos::finalize()``. This
includes the views stored in ``KokkosDataArray``, and thus any
``TensorBlock``/``TensorMap`` containing them.

.. doxygenclass:: metatensor::KokkosDataArray
    :members: KokkosDataArray, operator=, data, strides, is_contiguous, view, from_mts_array

.. _Kokkos: https://kokkos.org/
