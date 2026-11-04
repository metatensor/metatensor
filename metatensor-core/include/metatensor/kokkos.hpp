#ifndef METATENSOR_KOKKOS_HPP
#define METATENSOR_KOKKOS_HPP

/// This file provides `metatensor::KokkosDataArray`, an implementation of
/// `metatensor::DataArrayBase` using `Kokkos::View` to store the data.
///
/// This file is NOT included by `<metatensor.hpp>`, and must be explicitly
/// included by users who want to use Kokkos. Metatensor does not provide Kokkos
/// itself, users should make sure that Kokkos headers are available in the
/// include path (e.g. by linking to the `Kokkos::kokkos` CMake target).

#include <Kokkos_Core.hpp>

#if !defined(KOKKOS_VERSION)
    #error "metatensor/kokkos.hpp could not determine the version of Kokkos (KOKKOS_VERSION is not defined)"
#endif



#define MTS_KK_STRINGIFY_IMPL(x) #x
#define MTS_KK_STRINGIFY(x) MTS_KK_STRINGIFY_IMPL(x)

#if KOKKOS_VERSION < 40000
    #define MTS_KK_VERSION_ERROR                                                      \
        "metatensor/kokkos.hpp requires Kokkos 4.0.0 or later, but an older version " \
        "of Kokkos was found (KOKKOS_VERSION=" MTS_KK_STRINGIFY(KOKKOS_VERSION) ")"
    #if defined(_MSC_VER)
        #pragma message("error: " MTS_KK_VERSION_ERROR)
        #error "metatensor/kokkos.hpp requires Kokkos 4.0.0 or later"
    #else
        // `#pragma GCC error` expands macros, which `#error` does not do
        #pragma GCC error MTS_KK_VERSION_ERROR
    #endif
    #undef MTS_KK_VERSION_ERROR
#endif

#if KOKKOS_VERSION >= 60000
    #define MTS_KK_VERSION_WARNING                                                  \
        "metatensor/kokkos.hpp has only been tested with Kokkos 4.x and 5.x, but "  \
        "KOKKOS_VERSION=" MTS_KK_STRINGIFY(KOKKOS_VERSION) " was found. "           \
        "It might not work correctly."
    #if defined(_MSC_VER)
        #pragma message("warning: " MTS_KK_VERSION_WARNING)
    #else
        #pragma GCC warning MTS_KK_VERSION_WARNING
    #endif
    #undef MTS_KK_VERSION_WARNING
#endif

#if defined(KOKKOS_ENABLE_SYCL)
    // Kokkos can use SYCL to target NVIDIA and AMD GPUs (with
    // Kokkos_ENABLE_UNSUPPORTED_ARCHS=ON). The memory is then CUDA/HIP memory,
    // which we would need to export to DLPack as kDLCUDA/kDLROCM instead of
    // kDLOneAPI. This is not implemented, so we only support Intel GPUs with
    // SYCL. Use the CUDA or HIP backends of Kokkos for NVIDIA and AMD GPUs.
    #if KOKKOS_VERSION < 40100
        // KOKKOS_IMPL_ARCH_NVIDIA_GPU is only defined since Kokkos 4.1
        #error "metatensor/kokkos.hpp requires Kokkos 4.1.0 or later when using the SYCL backend"
    #endif

    #if defined(KOKKOS_IMPL_ARCH_NVIDIA_GPU)
        #error "metatensor/kokkos.hpp does not support the SYCL backend of Kokkos on NVIDIA GPUs, please use the CUDA backend instead"
    #endif

    #if defined(KOKKOS_ARCH_AMD_GPU)
        #error "metatensor/kokkos.hpp does not support the SYCL backend of Kokkos on AMD GPUs, please use the HIP backend instead"
    #endif
#endif

#include <cstdint>
#include <cstring>

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <type_traits>

#include "./errors.hpp"
#include "./arrays.hpp"

namespace metatensor {

template <typename T, typename MemorySpace>
class KokkosDataArray;

namespace details {
    /// Maximal number of dimensions supported by the device kernels in
    /// `KokkosDataArray`.
    constexpr size_t KOKKOS_MAX_NDIM = 16;

#if defined(KOKKOS_ENABLE_SYCL)
    // the SYCL backend moved out of `Kokkos::Experimental` in Kokkos 4.5
    #if KOKKOS_VERSION >= 40500
    using KokkosSYCL = Kokkos::SYCL;
    using KokkosSYCLDeviceUSMSpace = Kokkos::SYCLDeviceUSMSpace;
    using KokkosSYCLSharedUSMSpace = Kokkos::SYCLSharedUSMSpace;
    using KokkosSYCLHostUSMSpace = Kokkos::SYCLHostUSMSpace;
    #else
    using KokkosSYCL = Kokkos::Experimental::SYCL;
    using KokkosSYCLDeviceUSMSpace = Kokkos::Experimental::SYCLDeviceUSMSpace;
    using KokkosSYCLSharedUSMSpace = Kokkos::Experimental::SYCLSharedUSMSpace;
    using KokkosSYCLHostUSMSpace = Kokkos::Experimental::SYCLHostUSMSpace;
    #endif

    /// Get the DLPack `device_id` of the SYCL device used by Kokkos.
    inline int32_t kokkos_sycl_device_id() {
        auto device = KokkosSYCL().sycl_queue().get_device();
        auto all_devices = sycl::device::get_devices();
        for (size_t i = 0; i < all_devices.size(); i++) {
            if (all_devices[i] == device) {
                return static_cast<int32_t>(i);
            }
        }
        throw Error("could not find the SYCL device used by Kokkos in the list of all SYCL devices");
    }
#endif

    /// Information about how a Kokkos memory space maps to DLPack devices.
    ///
    /// This is only specialized for the memory spaces supported by
    /// `KokkosDataArray`.
    template <typename MemorySpace>
    struct KokkosMemorySpace {
        // `sizeof(MemorySpace) == 0` is always false, but depends on the
        // template parameter, so the assertion only fails if this template is
        // instantiated, i.e. when using an unsupported memory space.
        static_assert(
            sizeof(MemorySpace) == 0,
            "this Kokkos memory space is not supported by metatensor::KokkosDataArray"
        );
    };

    template <>
    struct KokkosMemorySpace<Kokkos::HostSpace> {
        static DLDevice device() { return {kDLCPU, 0}; }
    };

#if defined(KOKKOS_ENABLE_CUDA)
    template <>
    struct KokkosMemorySpace<Kokkos::CudaSpace> {
        static DLDevice device() { return {kDLCUDA, Kokkos::Cuda().cuda_device()}; }
    };

    template <>
    struct KokkosMemorySpace<Kokkos::CudaUVMSpace> {
        static DLDevice device() { return {kDLCUDAManaged, Kokkos::Cuda().cuda_device()}; }
    };

    template <>
    struct KokkosMemorySpace<Kokkos::CudaHostPinnedSpace> {
        static DLDevice device() { return {kDLCUDAHost, 0}; }
    };
#endif

#if defined(KOKKOS_ENABLE_HIP)
    template <>
    struct KokkosMemorySpace<Kokkos::HIPSpace> {
        static DLDevice device() { return {kDLROCM, Kokkos::HIP().hip_device()}; }
    };

    template <>
    struct KokkosMemorySpace<Kokkos::HIPHostPinnedSpace> {
        static DLDevice device() { return {kDLROCMHost, 0}; }
    };

    template <>
    struct KokkosMemorySpace<Kokkos::HIPManagedSpace> {
        // DLPack does not have a separate device type for HIP managed memory
        static DLDevice device() { return {kDLROCM, Kokkos::HIP().hip_device()}; }
    };
#endif

#if defined(KOKKOS_ENABLE_SYCL)
    template <>
    struct KokkosMemorySpace<KokkosSYCLDeviceUSMSpace> {
        static DLDevice device() { return {kDLOneAPI, kokkos_sycl_device_id()}; }
    };

    template <>
    struct KokkosMemorySpace<KokkosSYCLSharedUSMSpace> {
        // DLPack does not have a separate device type for SYCL shared memory
        static DLDevice device() { return {kDLOneAPI, kokkos_sycl_device_id()}; }
    };

    template <>
    struct KokkosMemorySpace<KokkosSYCLHostUSMSpace> {
        // SYCL host USM is accessible from the CPU
        static DLDevice device() { return {kDLCPU, 0}; }
    };
#endif

    inline std::string dldevice_to_string(DLDevice device) {
        std::string device_type = std::to_string(static_cast<int>(device.device_type));
        if (device.device_type == kDLCPU) {
            device_type = "kDLCPU";
        } else if (device.device_type == kDLCUDA) {
            device_type = "kDLCUDA";
        } else if (device.device_type == kDLCUDAManaged) {
            device_type = "kDLCUDAManaged";
        } else if (device.device_type == kDLCUDAHost) {
            device_type = "kDLCUDAHost";
        } else if (device.device_type == kDLROCM) {
            device_type = "kDLROCM";
        } else if (device.device_type == kDLROCMHost) {
            device_type = "kDLROCMHost";
        } else if (device.device_type == kDLOneAPI) {
            device_type = "kDLOneAPI";
        } else if (device.device_type == kDLOpenCL) {
            device_type = "kDLOpenCL";
        } else if (device.device_type == kDLVulkan) {
            device_type = "kDLVulkan";
        } else if (device.device_type == kDLMetal) {
            device_type = "kDLMetal";
        } else if (device.device_type == kDLExtDev) {
            device_type = "kDLExtDev";
        } else if (device.device_type == kDLWebGPU) {
            device_type = "kDLWebGPU";
        }

        return "(" + device_type + ", " + std::to_string(device.device_id) + ")";
    }

    /// Check that the memory space `MemorySpace` corresponds to exactly
    /// `device`, and call `function(MemorySpace{})`.
    template <typename MemorySpace, typename Function>
    auto call_with_kokkos_memory_space(DLDevice device, Function&& function) {
        auto expected = KokkosMemorySpace<MemorySpace>::device();
        if (expected.device_type != device.device_type || expected.device_id != device.device_id) {
            throw Error(
                "requested device " + dldevice_to_string(device) +
                " does not match the device used by Kokkos " +
                dldevice_to_string(expected)
            );
        }
        return function(MemorySpace{});
    }

    /// Find the Kokkos memory space corresponding to the given DLPack
    /// `device`, and call `function(MemorySpace{})`. This throws an error if
    /// there is no memory space matching the `device`.
    template <typename Function>
    auto with_kokkos_memory_space(DLDevice device, Function&& function) {
        if (device.device_type == kDLCPU) {
            return call_with_kokkos_memory_space<Kokkos::HostSpace>(device, function);
        }
#if defined(KOKKOS_ENABLE_CUDA)
        if (device.device_type == kDLCUDA) {
            return call_with_kokkos_memory_space<Kokkos::CudaSpace>(device, function);
        } else if (device.device_type == kDLCUDAManaged) {
            return call_with_kokkos_memory_space<Kokkos::CudaUVMSpace>(device, function);
        } else if (device.device_type == kDLCUDAHost) {
            return call_with_kokkos_memory_space<Kokkos::CudaHostPinnedSpace>(device, function);
        }
#endif
#if defined(KOKKOS_ENABLE_HIP)
        if (device.device_type == kDLROCM) {
            return call_with_kokkos_memory_space<Kokkos::HIPSpace>(device, function);
        } else if (device.device_type == kDLROCMHost) {
            return call_with_kokkos_memory_space<Kokkos::HIPHostPinnedSpace>(device, function);
        }
#endif
#if defined(KOKKOS_ENABLE_SYCL)
        if (device.device_type == kDLOneAPI) {
            return call_with_kokkos_memory_space<KokkosSYCLDeviceUSMSpace>(device, function);
        }
#endif
        throw Error(
            "there is no Kokkos memory space corresponding to DLPack device " +
            dldevice_to_string(device) + " in this build of Kokkos"
        );
    }

    /// Tag type used to pass a C++ type to a generic lambda
    template <typename T>
    struct TypeTag { using type = T; };

    /// Call `function(TypeTag<T>{})` with the C++ type `T` corresponding to the
    /// given DLPack `dtype`.
    template <typename Function>
    auto with_dlpack_dtype(DLDataType dtype, Function&& function) {
        if (dtype.lanes != 1) {
            throw Error(
                "only DLPack data types with lanes == 1 are supported, got lanes=" +
                std::to_string(dtype.lanes)
            );
        }

        if (dtype.code == kDLFloat && dtype.bits == 64) {
            return function(TypeTag<double>{});
        } else if (dtype.code == kDLFloat && dtype.bits == 32) {
            return function(TypeTag<float>{});
        } else if (dtype.code == kDLInt && dtype.bits == 8) {
            return function(TypeTag<int8_t>{});
        } else if (dtype.code == kDLInt && dtype.bits == 16) {
            return function(TypeTag<int16_t>{});
        } else if (dtype.code == kDLInt && dtype.bits == 32) {
            return function(TypeTag<int32_t>{});
        } else if (dtype.code == kDLInt && dtype.bits == 64) {
            return function(TypeTag<int64_t>{});
        } else if (dtype.code == kDLUInt && dtype.bits == 8) {
            return function(TypeTag<uint8_t>{});
        } else if (dtype.code == kDLUInt && dtype.bits == 16) {
            return function(TypeTag<uint16_t>{});
        } else if (dtype.code == kDLUInt && dtype.bits == 32) {
            return function(TypeTag<uint32_t>{});
        } else if (dtype.code == kDLUInt && dtype.bits == 64) {
            return function(TypeTag<uint64_t>{});
        } else if (dtype.code == kDLBool && dtype.bits == 8) {
            return function(TypeTag<bool>{});
        } else {
            throw Error(
                "unsupported DLPack data type: code=" + std::to_string(dtype.code) +
                " bits=" + std::to_string(dtype.bits)
            );
        }
    }


    /// Information about a Kokkos `DataType` such as `double**[3]`: total
    /// number of dimensions, number of dynamic dimensions, and the static
    /// extents.
    ///
    /// Kokkos puts all dynamic dimensions (`*` or `[]`) before the static ones
    /// (`[N]`), so `static_extents` writes the static extents in order,
    /// without any entry for the dynamic dimensions.
    template <typename DataType>
    struct kokkos_data_rank {
        static constexpr size_t value = 0;
        static constexpr size_t dynamic = 0;
        static void static_extents(size_t*) {}
    };

    template <typename DataType>
    struct kokkos_data_rank<DataType*> {
        static constexpr size_t value = 1 + kokkos_data_rank<DataType>::value;
        static constexpr size_t dynamic = 1 + kokkos_data_rank<DataType>::dynamic;
        static void static_extents(size_t* extents) {
            kokkos_data_rank<DataType>::static_extents(extents);
        }
    };

    template <typename DataType>
    struct kokkos_data_rank<DataType[]> {
        static constexpr size_t value = 1 + kokkos_data_rank<DataType>::value;
        static constexpr size_t dynamic = 1 + kokkos_data_rank<DataType>::dynamic;
        static void static_extents(size_t* extents) {
            kokkos_data_rank<DataType>::static_extents(extents);
        }
    };

    template <typename DataType, size_t N>
    struct kokkos_data_rank<DataType[N]> {
        static constexpr size_t value = 1 + kokkos_data_rank<DataType>::value;
        static constexpr size_t dynamic = kokkos_data_rank<DataType>::dynamic;
        static void static_extents(size_t* extents) {
            extents[0] = N;
            kokkos_data_rank<DataType>::static_extents(extents + 1);
        }
    };

    /// Check if an array with the given `shape` and `strides` is contiguous
    /// in column-major order (i.e. compatible with `Kokkos::LayoutLeft`).
    inline bool is_column_major_contiguous(const std::vector<uintptr_t>& shape, const std::vector<int64_t>& strides) {
        int64_t expected_stride = 1;
        for (size_t dim = 0; dim < shape.size(); dim++) {
            if (shape[dim] != 1 && strides[dim] != expected_stride) {
                return false;
            }
            expected_stride *= static_cast<int64_t>(shape[dim]);
        }
        return true;
    }

    /// Try to find the strides corresponding to reshaping an array with
    /// `old_shape` and `old_strides` to `new_shape`, without copying the
    /// data. This works by grouping together consecutive dimensions in the
    /// old and new shape with the same total size, and checking that the old
    /// dimensions in each group can be merged (i.e. they are contiguous with
    /// respect to each other).
    ///
    /// This returns `true` and sets `new_strides` on success, and `false` if
    /// the data needs to be copied to be reshaped (for example when
    /// flattening a transposed array).
    inline bool reshape_strides(
        const std::vector<uintptr_t>& old_shape,
        const std::vector<int64_t>& old_strides,
        const std::vector<uintptr_t>& new_shape,
        std::vector<int64_t>& new_strides
    ) {
        if (details::product(new_shape) == 0) {
            // any strides are valid for empty arrays
            new_strides = details::contiguous_strides(new_shape);
            return true;
        }

        // dimensions of size 1 can have any stride, ignore them
        auto shape = std::vector<uintptr_t>();
        auto strides = std::vector<int64_t>();
        for (size_t dim = 0; dim < old_shape.size(); dim++) {
            if (old_shape[dim] != 1) {
                shape.push_back(old_shape[dim]);
                strides.push_back(old_strides[dim]);
            }
        }

        new_strides.assign(new_shape.size(), 1);

        size_t old_start = 0;
        size_t new_start = 0;
        while (old_start < shape.size() && new_start < new_shape.size()) {
            // find the smallest groups of dimensions [old_start, old_end) and
            // [new_start, new_end) with the same total size
            auto old_end = old_start + 1;
            auto new_end = new_start + 1;
            auto old_size = shape[old_start];
            auto new_size = new_shape[new_start];
            while (old_size != new_size) {
                if (new_size < old_size) {
                    new_size *= new_shape[new_end];
                    new_end += 1;
                } else {
                    old_size *= shape[old_end];
                    old_end += 1;
                }
            }

            // the old dimensions in this group must be contiguous with
            // respect to each other
            for (auto dim = old_start; dim + 1 < old_end; dim++) {
                if (strides[dim] != strides[dim + 1] * static_cast<int64_t>(shape[dim + 1])) {
                    return false;
                }
            }

            // the new dimensions in this group are row-major contiguous,
            // starting from the stride of the last old dimension
            new_strides[new_end - 1] = strides[old_end - 1];
            for (auto dim = new_end - 1; dim > new_start; dim--) {
                new_strides[dim - 1] = new_strides[dim] * static_cast<int64_t>(new_shape[dim]);
            }

            old_start = old_end;
            new_start = new_end;
        }

        // the remaining new dimensions all have size 1, their stride does not
        // matter. We still set them to a contiguous value, to keep
        // `is_contiguous()` true for contiguous arrays.
        for (auto dim = new_start; dim < new_shape.size(); dim++) {
            new_strides[dim] = dim == 0 ? 1 : new_strides[dim - 1];
        }

        return true;
    }

    /// Flat, contiguous Kokkos view
    template <typename T, typename MemorySpace>
    using KokkosFlatView = Kokkos::View<T*, Kokkos::LayoutRight, MemorySpace>;

    /// Allocate a new flat view with `size` elements, without initializing it
    template <typename T, typename MemorySpace>
    KokkosFlatView<T, MemorySpace> kokkos_allocate(size_t size) {
        return KokkosFlatView<T, MemorySpace>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("metatensor::KokkosDataArray")),
            size
        );
    }

    /// Fixed-size array used to send shapes and strides to device kernels
    using KokkosIndexArray = Kokkos::Array<int64_t, KOKKOS_MAX_NDIM>;

    template <typename Shape>
    KokkosIndexArray to_kokkos_index_array(const Shape& values) {
        if (values.size() > KOKKOS_MAX_NDIM) {
            throw Error(
                "KokkosDataArray only supports arrays with up to " +
                std::to_string(KOKKOS_MAX_NDIM) + " dimensions, got " +
                std::to_string(values.size())
            );
        }

        auto result = KokkosIndexArray();
        for (size_t i = 0; i < KOKKOS_MAX_NDIM; i++) {
            result[i] = 0;
        }
        for (size_t i = 0; i < values.size(); i++) {
            result[i] = static_cast<int64_t>(values[i]);
        }
        return result;
    }

    /// Copy data from the strided array starting at `input` (with the given
    /// `shape` and `strides`, in number of elements) to the contiguous
    /// row-major `output`. `input` must be accessible from the execution space
    /// associated with `MemorySpace`.
    ///
    /// This function waits for the copy to finish before returning.
    template <typename T, typename MemorySpace>
    void kokkos_strided_copy(
        const T* input,
        const std::vector<uintptr_t>& shape,
        const std::vector<int64_t>& strides,
        const KokkosFlatView<T, MemorySpace>& output
    ) {
        using execution_space = typename MemorySpace::execution_space;

        auto size = static_cast<int64_t>(details::product(shape));
        assert(static_cast<int64_t>(output.extent(0)) == size);
        if (size == 0) {
            return;
        }

        auto ndim = shape.size();
        auto kokkos_shape = to_kokkos_index_array(shape);
        auto kokkos_strides = to_kokkos_index_array(strides);

        Kokkos::parallel_for(
            "metatensor::KokkosDataArray::strided_copy",
            Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int64_t>>(0, size),
            KOKKOS_LAMBDA(int64_t i) {
                auto remaining = i;
                int64_t offset = 0;
                for (size_t d = ndim; d > 0; d--) {
                    auto dim = d - 1;
                    auto index = remaining % kokkos_shape[dim];
                    remaining = remaining / kokkos_shape[dim];
                    offset += index * kokkos_strides[dim];
                }
                output(i) = input[offset];
            }
        );
        execution_space().fence("metatensor::KokkosDataArray::strided_copy");
    }

    /// DLPack context keeping the data alive while the corresponding
    /// `DLManagedTensorVersioned` is in use.
    struct KokkosDLPackContext: DLPackContextBase {
        std::shared_ptr<void> keepalive;
    };

    mts_data_origin_t kokkos_origin() {
        mts_data_origin_t origin = 0;
        details::check_status(mts_register_data_origin("metatensor::KokkosDataArray", &origin));
        return origin;
    }
} // namespace details


/// Implementation of `DataArrayBase` using Kokkos to store the data.
///
/// `T` is the type of the array elements, and `MemorySpace` the Kokkos memory
/// space where the data is allocated (defaults to the memory space of
/// `Kokkos::DefaultExecutionSpace`).
///
/// The data can have any memory layout: the array stores a pointer to the
/// first element, together with the shape and the strides (in number of
/// elements) of the array. This allows to wrap any `Kokkos::View` (with
/// `Kokkos::LayoutRight`, `Kokkos::LayoutLeft`, `Kokkos::LayoutStride`, ...)
/// without copying the data.
///
/// This class is only available when including `<metatensor/kokkos.hpp>`, and
/// requires Kokkos to be available in the include path.
///
/// **WARNING**: Kokkos requires all views to be destroyed before calling
/// `Kokkos::finalize()`. This includes the views stored inside
/// `KokkosDataArray`, and thus all `TensorBlock`/`TensorMap` containing
/// `KokkosDataArray`, as well as any DLPack tensors created from them.
template <typename T, typename MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class KokkosDataArray: public DataArrayBase {
public:
    static_assert(std::is_arithmetic_v<T>, "KokkosDataArray requires an arithmetic type");
    // instantiate `KokkosMemorySpace` early, to get a clear error message for
    // unsupported memory spaces before any other error coming from Kokkos.
    static_assert(sizeof(details::KokkosMemorySpace<MemorySpace>) != 0);

    /// Type of a flat, contiguous Kokkos view in this memory space
    using flat_view_t = details::KokkosFlatView<T, MemorySpace>;

    /// Create a `KokkosDataArray` from an existing n-dimensional
    /// `Kokkos::View`, with any layout. The shape and strides of the array are
    /// taken from the view. The data is shared with the view (no copy is
    /// made), and the view is kept alive by this `KokkosDataArray`.
    template <typename DataType, typename... Properties>
    explicit KokkosDataArray(Kokkos::View<DataType, Properties...> view) {
        using input_view_t = Kokkos::View<DataType, Properties...>;
        using value_t = typename input_view_t::value_type;
        static_assert(
            std::is_same_v<value_t, T>,
            "the view value type must match the KokkosDataArray type (const views are not supported)"
        );
        static_assert(
            std::is_same_v<typename input_view_t::memory_space, MemorySpace>,
            "the view memory space must match the KokkosDataArray memory space"
        );

        constexpr auto rank = details::kokkos_data_rank<DataType>::value;
        // `if constexpr` avoids warnings about `dim < 0` for rank 0 views
        if constexpr (rank > 0) {
            for (size_t dim = 0; dim < rank; dim++) {
                shape_.push_back(static_cast<uintptr_t>(view.extent(dim)));
                strides_.push_back(static_cast<int64_t>(view.stride(dim)));
            }
        }

        data_ = view.data();
        keepalive_ = std::make_shared<input_view_t>(std::move(view));
    }

    /// Create a `KokkosDataArray` with the given `shape`, and all elements set
    /// to `value`. The data is allocated contiguously, in row-major order.
    KokkosDataArray(std::vector<uintptr_t> shape, T value = T{}):
        shape_(std::move(shape))
    {
        auto view = details::kokkos_allocate<T, MemorySpace>(details::product(shape_));
        Kokkos::deep_copy(view, value);
        this->set_contiguous_storage(std::move(view));
    }

    /// Create a `KokkosDataArray` with the given `shape`, using `data` as
    /// storage. The data is interpreted as a contiguous row-major array, and
    /// is shared with the caller (no copy is made).
    KokkosDataArray(std::vector<uintptr_t> shape, flat_view_t data):
        shape_(std::move(shape))
    {
        if (data.extent(0) != details::product(shape_)) {
            throw Error("the shape and size of the data don't match in KokkosDataArray");
        }
        this->set_contiguous_storage(std::move(data));
    }

    ~KokkosDataArray() override = default;

    /// KokkosDataArray can be copy-constructed. The data is shared between
    /// the copies.
    KokkosDataArray(const KokkosDataArray&) = default;
    /// KokkosDataArray can be copy-assigned. The data is shared between the
    /// copies.
    KokkosDataArray& operator=(const KokkosDataArray&) = default;
    /// KokkosDataArray can be move-constructed
    KokkosDataArray(KokkosDataArray&&) noexcept = default;
    /// KokkosDataArray can be move-assigned
    KokkosDataArray& operator=(KokkosDataArray&&) noexcept = default;

    mts_data_origin_t origin() const override {
        static auto origin = details::kokkos_origin();
        return origin;
    }

    DLDevice device() const override {
        return details::KokkosMemorySpace<MemorySpace>::device();
    }

    DLDataType dtype() const override {
        return details::dtype_of<T>();
    }

    const std::vector<uintptr_t>& shape() const & override {
        return shape_;
    }

    /// Change the shape of this array. Whenever possible, this only changes
    /// the shape and strides without copying the data. When the new shape
    /// can not be represented with strides over the existing data (for
    /// example when flattening a transposed array), the data is first copied
    /// to a new contiguous array.
    void reshape(const std::vector<uintptr_t>& shape) override {
        if (details::product(shape_) != details::product(shape)) {
            throw Error("invalid shape in reshape");
        }

        auto new_strides = std::vector<int64_t>();
        if (details::reshape_strides(shape_, strides_, shape, new_strides)) {
            shape_ = shape;
            strides_ = std::move(new_strides);
        } else {
            auto contiguous = this->contiguous_copy();
            shape_ = shape;
            this->set_contiguous_storage(std::move(contiguous));
        }
    }

    /// Swap the axes `axis_1` and `axis_2` of this array. This only changes
    /// the shape and strides, the data is not copied.
    void swap_axes(uintptr_t axis_1, uintptr_t axis_2) override {
        if (axis_1 >= shape_.size() || axis_2 >= shape_.size()) {
            throw Error("out of bounds axis in KokkosDataArray::swap_axes");
        }

        std::swap(shape_[axis_1], shape_[axis_2]);
        std::swap(strides_[axis_1], strides_[axis_2]);
    }

    /// Copy this array to the given `device`. The new array is always
    /// contiguous in row-major order.
    std::unique_ptr<DataArrayBase> copy(DLDevice device) const override {
        auto contiguous = this->contiguous_view();
        return details::with_kokkos_memory_space(device, [&](auto space) {
            using space_t = decltype(space);
            auto new_data = details::kokkos_allocate<T, space_t>(contiguous.extent(0));
            Kokkos::deep_copy(new_data, contiguous);
            return std::unique_ptr<DataArrayBase>(
                new KokkosDataArray<T, space_t>(shape_, std::move(new_data))
            );
        });
    }

    std::unique_ptr<DataArrayBase> create(
        std::vector<uintptr_t> shape,
        MtsArray fill_value
    ) const override {
        DLDevice cpu_device = {kDLCPU, 0};
        DLPackVersion max_version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        auto fill_value_dlpack = fill_value.as_dlpack_array<T>(cpu_device, nullptr, max_version);

        if (!fill_value_dlpack.shape().empty()) {
            throw Error("`fill_value` must be a single scalar");
        }

        auto scalar = fill_value_dlpack.data()[0];
        return std::unique_ptr<DataArrayBase>(new KokkosDataArray(std::move(shape), scalar));
    }

    void move_data(
        const DataArrayBase& input,
        std::vector<mts_data_movement_t> moves
    ) override {
        using execution_space = typename MemorySpace::execution_space;

        const auto* input_array = dynamic_cast<const KokkosDataArray*>(&input);
        if (input_array == nullptr) {
            throw Error(
                "`input` in KokkosDataArray::move_data must be a KokkosDataArray "
                "with the same data type and memory space"
            );
        }

        if (moves.empty()) {
            return;
        }

        const auto& input_shape = input_array->shape();
        const auto& input_strides = input_array->strides();
        const auto& output_shape = this->shape();
        const auto& output_strides = this->strides();

        if (input_shape.size() != output_shape.size() || output_shape.size() < 2) {
            throw Error(
                "invalid shapes in KokkosDataArray::move_data: input and output "
                "must have the same number of dimensions, and at least 2 dimensions"
            );
        }

        auto property_dim = output_shape.size() - 1;

        // Product of all the components dimensions
        int64_t n_components = 1;
        for (size_t i = 1; i < property_dim; i++) {
            n_components *= static_cast<int64_t>(output_shape[i]);
        }

        auto shape = details::to_kokkos_index_array(output_shape);
        auto in_strides = details::to_kokkos_index_array(input_strides);
        auto out_strides = details::to_kokkos_index_array(output_strides);

        auto host_moves = Kokkos::View<const mts_data_movement_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            moves.data(), moves.size()
        );
        auto device_moves = Kokkos::create_mirror_view_and_copy(MemorySpace(), host_moves);

        auto* output_ptr = this->data();
        const auto* input_ptr = input_array->data();
        using policy_t = Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2>, Kokkos::IndexType<int64_t>>;
        Kokkos::parallel_for(
            "metatensor::KokkosDataArray::move_data",
            policy_t({0, 0}, {static_cast<int64_t>(moves.size()), n_components}),
            KOKKOS_LAMBDA(int64_t move_i, int64_t component) {
                const auto& move = device_moves(move_i);

                auto input_offset = static_cast<int64_t>(move.sample_in) * in_strides[0];
                auto output_offset = static_cast<int64_t>(move.sample_out) * out_strides[0];

                // decompose the linear component index into one index for
                // each of the component dimensions (in row-major order)
                auto remaining = component;
                for (size_t d = property_dim; d > 1; d--) {
                    auto dim = d - 1;
                    auto index = remaining % shape[dim];
                    remaining = remaining / shape[dim];
                    input_offset += index * in_strides[dim];
                    output_offset += index * out_strides[dim];
                }

                input_offset += static_cast<int64_t>(move.properties_start_in) * in_strides[property_dim];
                output_offset += static_cast<int64_t>(move.properties_start_out) * out_strides[property_dim];

                auto length = static_cast<int64_t>(move.properties_length);
                for (int64_t p = 0; p < length; p++) {
                    auto output_idx = output_offset + p * out_strides[property_dim];
                    auto input_idx = input_offset + p * in_strides[property_dim];
                    output_ptr[output_idx] = input_ptr[input_idx];
                }
            }
        );
        execution_space().fence("metatensor::KokkosDataArray::move_data");
    }

    /// Export this array as a DLPack tensor. If `device` is the device of this
    /// array, the data is shared with the DLPack tensor, keeping the current
    /// strides. Otherwise, the data is copied to a new contiguous array on the
    /// requested device.
    DLManagedTensorVersioned* as_dlpack(
        DLDevice device,
        const int64_t* stream,
        DLPackVersion max_version
    ) override {
        if (max_version.major != DLPACK_MAJOR_VERSION) {
            throw Error(
                "invalid `max_version` in KokkosDataArray::as_dlpack: "
                "we got v" + std::to_string(max_version.major) + "." +
                std::to_string(max_version.minor) + ", but we support v" +
                std::to_string(DLPACK_MAJOR_VERSION) + "." +
                std::to_string(DLPACK_MINOR_VERSION)
            );
        }

        if (device.device_type == kDLCPU && stream != nullptr) {
            throw Error("`stream` must be null for CPU data");
        }

        DLManagedTensorVersioned* managed = nullptr;
        auto current_device = this->device();
        if (device.device_type == current_device.device_type && device.device_id == current_device.device_id) {
            managed = KokkosDataArray::export_dlpack(data_, shape_, strides_, keepalive_, device);
        } else {
            auto contiguous = this->contiguous_view();
            managed = details::with_kokkos_memory_space(device, [&](auto space) {
                using space_t = decltype(space);
                auto new_data = details::kokkos_allocate<T, space_t>(contiguous.extent(0));
                Kokkos::deep_copy(new_data, contiguous);

                auto* data = new_data.data();
                return KokkosDataArray::export_dlpack(
                    data,
                    shape_,
                    details::contiguous_strides(shape_),
                    std::make_shared<details::KokkosFlatView<T, space_t>>(std::move(new_data)),
                    details::KokkosMemorySpace<space_t>::device()
                );
            });
        }

        // `stream == -1` means that no synchronization should happen.
        if (stream == nullptr || *stream != -1) {
            Kokkos::fence("metatensor::KokkosDataArray::as_dlpack");
        }

        return managed;
    }

    /// Create a new `KokkosDataArray` from a DLPack tensor. The data is
    /// shared with the DLPack tensor (no copy is made), keeping the strides of
    /// the tensor. The memory space of the new array is determined by the
    /// device of the DLPack tensor.
    std::unique_ptr<DataArrayBase> from_dlpack(DLManagedTensorVersioned* dl_managed_tensor) const override {
        if (dl_managed_tensor == nullptr) {
            throw Error("dlpack tensor is null in KokkosDataArray::from_dlpack");
        }

        // take ownership of the tensor, the deleter will be called when the
        // last reference to `owner` goes away.
        auto owner = std::shared_ptr<DLManagedTensorVersioned>(
            dl_managed_tensor,
            [](DLManagedTensorVersioned* tensor) {
                if (tensor->deleter != nullptr) {
                    tensor->deleter(tensor);
                }
            }
        );

        if (dl_managed_tensor->version.major != DLPACK_MAJOR_VERSION) {
            throw Error(
                "DLPack version mismatch in KokkosDataArray::from_dlpack: expected major " +
                std::to_string(DLPACK_MAJOR_VERSION) + ", got " +
                std::to_string(dl_managed_tensor->version.major) + "." +
                std::to_string(dl_managed_tensor->version.minor)
            );
        }

        const auto& dl_tensor = dl_managed_tensor->dl_tensor;

        auto shape = std::vector<uintptr_t>();
        for (int32_t i = 0; i < dl_tensor.ndim; i++) {
            shape.push_back(static_cast<uintptr_t>(dl_tensor.shape[i]));
        }

        auto strides = std::vector<int64_t>();
        if (dl_tensor.strides == nullptr) {
            strides = details::contiguous_strides(shape);
        } else {
            strides.assign(dl_tensor.strides, dl_tensor.strides + dl_tensor.ndim);
        }

        return details::with_dlpack_dtype(dl_tensor.dtype, [&](auto tag) {
            using value_t = typename decltype(tag)::type;
            return details::with_kokkos_memory_space(dl_tensor.device, [&](auto space) {
                using space_t = decltype(space);
                using new_array_t = KokkosDataArray<value_t, space_t>;

                auto* data = reinterpret_cast<value_t*>(
                    static_cast<char*>(dl_tensor.data) + dl_tensor.byte_offset
                );

                return std::unique_ptr<DataArrayBase>(new new_array_t(
                    shape, strides, data, owner
                ));
            });
        });
    }

    /// Get a pointer to the first element of this array. The pointer is in
    /// `MemorySpace`, and the data should be accessed using `shape()` and
    /// `strides()`.
    T* data() const {
        return data_;
    }

    /// Get the strides of this array, in number of elements.
    const std::vector<int64_t>& strides() const {
        return strides_;
    }

    /// Check if this array is contiguous in row-major order.
    bool is_contiguous() const {
        return details::is_row_major_contiguous(shape_, strides_);
    }

    /// Get an unmanaged n-dimensional Kokkos view of the data in this array.
    /// `DataType` should be the Kokkos data type of the view, e.g. `double**`
    /// for a 2-dimensional array of `double`. The returned view is only valid
    /// as long as this `KokkosDataArray` is alive.
    ///
    /// `Layout` can be `Kokkos::LayoutStride` (the default), which works for
    /// all arrays; or `Kokkos::LayoutRight`/`Kokkos::LayoutLeft`, which only
    /// work if the data is contiguous in row-major/column-major order.
    ///
    /// This function throws an error if the rank or the static extents of
    /// `DataType` do not match the shape of this array, or if the strides of
    /// this array are not compatible with `Layout`.
    template <typename DataType, typename Layout = Kokkos::LayoutStride>
    Kokkos::View<DataType, Layout, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> view() const {
        using result_t = Kokkos::View<DataType, Layout, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        static_assert(
            std::is_same_v<typename result_t::non_const_value_type, T>,
            "the view value type must match the KokkosDataArray type"
        );
        static_assert(
            std::is_same_v<Layout, Kokkos::LayoutStride> ||
            std::is_same_v<Layout, Kokkos::LayoutRight> ||
            std::is_same_v<Layout, Kokkos::LayoutLeft>,
            "KokkosDataArray::view only supports LayoutStride, LayoutRight and LayoutLeft"
        );

        constexpr auto rank = details::kokkos_data_rank<DataType>::value;
        constexpr auto rank_dynamic = details::kokkos_data_rank<DataType>::dynamic;
        if (rank != shape_.size()) {
            throw Error(
                "invalid rank in KokkosDataArray::view: the array has " +
                std::to_string(shape_.size()) + " dimensions, but the requested view has " +
                std::to_string(rank)
            );
        }

        // static extents of `DataType`, 0 for dynamic dimensions
        auto static_extents = std::array<size_t, rank>();
        details::kokkos_data_rank<DataType>::static_extents(static_extents.data() + rank_dynamic);
        // `if constexpr` avoids warnings about `dim < 0` for rank 0 views
        if constexpr (rank_dynamic < rank) {
            for (size_t dim = rank_dynamic; dim < rank; dim++) {
                if (static_extents[dim] != shape_[dim]) {
                    throw Error(
                        "invalid static extent in KokkosDataArray::view: the array has size " +
                        std::to_string(shape_[dim]) + " for dimension " + std::to_string(dim) +
                        ", but the requested view has size " + std::to_string(static_extents[dim])
                    );
                }
            }
        }

        if constexpr (std::is_same_v<Layout, Kokkos::LayoutStride>) {
            for (auto stride: strides_) {
                if (stride < 0) {
                    throw Error(
                        "this array has negative strides, which are not supported by "
                        "Kokkos::LayoutStride in KokkosDataArray::view"
                    );
                }
            }
            return this->view_stride_impl<result_t>(std::make_index_sequence<2 * rank>());
        } else {
            if constexpr (std::is_same_v<Layout, Kokkos::LayoutRight>) {
                if (!this->is_contiguous()) {
                    throw Error(
                        "this array is not contiguous in row-major order, and can not "
                        "be viewed with Kokkos::LayoutRight in KokkosDataArray::view, "
                        "use Kokkos::LayoutStride instead"
                    );
                }
            } else {
                if (!details::is_column_major_contiguous(shape_, strides_)) {
                    throw Error(
                        "this array is not contiguous in column-major order, and can not "
                        "be viewed with Kokkos::LayoutLeft in KokkosDataArray::view, "
                        "use Kokkos::LayoutStride instead"
                    );
                }
            }
            return this->view_dynamic_impl<result_t>(std::make_index_sequence<rank_dynamic>());
        }
    }

    /// Extract a reference to `KokkosDataArray` out of an `MtsArray`.
    ///
    /// This function fails if the `MtsArray` does not contain a
    /// `KokkosDataArray` with the same `T` and `MemorySpace`.
    static KokkosDataArray& from_mts_array(MtsArray& array) {
        auto* base = KokkosDataArray::check_origin(array);
        auto* result = dynamic_cast<KokkosDataArray*>(base);
        if (result == nullptr) {
            throw Error(
                "this array is a metatensor::KokkosDataArray, but with a "
                "different data type or memory space"
            );
        }
        return *result;
    }

    /// Extract a const reference to `KokkosDataArray` out of an `MtsArray`.
    ///
    /// This function fails if the `MtsArray` does not contain a
    /// `KokkosDataArray` with the same `T` and `MemorySpace`.
    static const KokkosDataArray& from_mts_array(const MtsArray& array) {
        const auto* base = KokkosDataArray::check_origin(array);
        const auto* result = dynamic_cast<const KokkosDataArray*>(base);
        if (result == nullptr) {
            throw Error(
                "this array is a metatensor::KokkosDataArray, but with a "
                "different data type or memory space"
            );
        }
        return *result;
    }

private:
    template <typename, typename>
    friend class KokkosDataArray;

    /// Create a `KokkosDataArray` from raw data, with the given `shape` and
    /// `strides`. `keepalive` should keep the data alive.
    KokkosDataArray(
        std::vector<uintptr_t> shape,
        std::vector<int64_t> strides,
        T* data,
        std::shared_ptr<void> keepalive
    ):
        shape_(std::move(shape)),
        strides_(std::move(strides)),
        data_(data),
        keepalive_(std::move(keepalive))
    {}

    /// Use the contiguous `view` as storage for this array, with the current
    /// `shape_`.
    void set_contiguous_storage(flat_view_t view) {
        strides_ = details::contiguous_strides(shape_);
        data_ = view.data();
        keepalive_ = std::make_shared<flat_view_t>(std::move(view));
    }

    /// Copy the data in this array to a new contiguous row-major view.
    flat_view_t contiguous_copy() const {
        auto view = details::kokkos_allocate<T, MemorySpace>(details::product(shape_));
        details::kokkos_strided_copy<T, MemorySpace>(data_, shape_, strides_, view);
        return view;
    }

    /// Get a contiguous row-major view of the data in this array. This is
    /// either an unmanaged view of the existing data (if it is already
    /// contiguous), or a new copy of the data.
    flat_view_t contiguous_view() const {
        if (this->is_contiguous()) {
            return flat_view_t(data_, details::product(shape_));
        } else {
            return this->contiguous_copy();
        }
    }

    template <typename View, size_t... I>
    View view_dynamic_impl(std::index_sequence<I...>) const {
        return View(data_, static_cast<size_t>(shape_[I])...);
    }

    template <typename View, size_t... I>
    View view_stride_impl(std::index_sequence<I...>) const {
        // arguments to `Kokkos::LayoutStride`: N0, S0, N1, S1, ...
        auto args = std::array<size_t, sizeof...(I)>();
        for (size_t dim = 0; dim < shape_.size(); dim++) {
            args[2 * dim] = static_cast<size_t>(shape_[dim]);
            args[2 * dim + 1] = static_cast<size_t>(strides_[dim]);
        }
        return View(data_, Kokkos::LayoutStride(args[I]...));
    }

    static DataArrayBase* check_origin(const MtsArray& array) {
        auto origin = array.origin();

        std::array<char, 64> buffer = {0};
        auto status = mts_get_data_origin(origin, buffer.data(), buffer.size());
        if (status != MTS_SUCCESS || std::string(buffer.data()) != "metatensor::KokkosDataArray") {
            throw Error("this array is not a metatensor::KokkosDataArray");
        }

        return static_cast<DataArrayBase*>(array.as_mts_array_t().ptr);
    }

    /// Export data as a DLPack tensor with the given shape and strides. The
    /// returned tensor keeps `keepalive` alive.
    static DLManagedTensorVersioned* export_dlpack(
        T* data,
        const std::vector<uintptr_t>& shape,
        const std::vector<int64_t>& strides,
        std::shared_ptr<void> keepalive,
        DLDevice device
    ) {
        auto ctx = std::make_unique<details::KokkosDLPackContext>();
        ctx->shape.resize(shape.size());
        std::transform(shape.begin(), shape.end(), ctx->shape.begin(),
                       [](uintptr_t s) { return static_cast<int64_t>(s); });
        ctx->strides = strides;
        ctx->keepalive = std::move(keepalive);

        auto managed = std::make_unique<DLManagedTensorVersioned>();
        managed->version.major = DLPACK_MAJOR_VERSION;
        managed->version.minor = DLPACK_MINOR_VERSION;
        managed->flags = 0;
        managed->deleter = details::DLPackDeleter;

        auto& tensor = managed->dl_tensor;
        tensor.device = device;
        tensor.dtype = details::dtype_of<T>();
        tensor.ndim = static_cast<int32_t>(shape.size());
        tensor.byte_offset = 0;
        tensor.data = static_cast<void*>(data);
        if (tensor.ndim == 0) {
            tensor.shape = nullptr;
            tensor.strides = nullptr;
        } else {
            tensor.shape = ctx->shape.data();
            tensor.strides = ctx->strides.data();
        }

        managed->manager_ctx = static_cast<void*>(ctx.release());
        return managed.release();
    }

    /// Shape of the array
    std::vector<uintptr_t> shape_;
    /// Strides of the array, in number of elements
    std::vector<int64_t> strides_;
    /// Pointer to the first element of the array
    T* data_ = nullptr;
    /// Type-erased object keeping the data pointed to by `data_` alive (Kokkos
    /// view, DLPack tensor, ...)
    std::shared_ptr<void> keepalive_;
};

/// Deduction guide for creating a `KokkosDataArray` from a `Kokkos::View`
template <typename DataType, typename... Properties>
KokkosDataArray(Kokkos::View<DataType, Properties...>) -> KokkosDataArray<
    typename Kokkos::View<DataType, Properties...>::value_type,
    typename Kokkos::View<DataType, Properties...>::memory_space
>;

} // namespace metatensor

#endif
