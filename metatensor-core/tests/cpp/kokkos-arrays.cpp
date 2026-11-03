#define CATCH_CONFIG_EXTERNAL_INTERFACES
#include <catch.hpp>

// Kokkos headers are not warning-free with -Weverything
#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Weverything"
#endif
#include <Kokkos_Core.hpp>
#if defined(__clang__)
    #pragma clang diagnostic pop
#endif

#include <metatensor.hpp>
#include <metatensor/kokkos.hpp>

using namespace metatensor;

/// Initialize and finalize Kokkos around all the tests
struct KokkosListener: Catch::TestEventListenerBase {
    using TestEventListenerBase::TestEventListenerBase;

    void testRunStarting(const Catch::TestRunInfo&) override {
        Kokkos::initialize();
    }

    void testRunEnded(const Catch::TestRunStats&) override {
        Kokkos::finalize();
    }
};
CATCH_REGISTER_LISTENER(KokkosListener)

using HostArray = KokkosDataArray<double, Kokkos::HostSpace>;

/// All the memory spaces supported by `KokkosDataArray` in the current build
/// of Kokkos. Tests using `TEMPLATE_LIST_TEST_CASE` with this list run once
/// for each memory space.
using MemorySpaces = std::tuple<
    Kokkos::HostSpace
#if defined(KOKKOS_ENABLE_CUDA)
    , Kokkos::CudaSpace
    , Kokkos::CudaUVMSpace
    , Kokkos::CudaHostPinnedSpace
#endif
#if defined(KOKKOS_ENABLE_HIP)
    , Kokkos::HIPSpace
    , Kokkos::HIPManagedSpace
    , Kokkos::HIPHostPinnedSpace
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    , details::KokkosSYCLDeviceUSMSpace
    , details::KokkosSYCLSharedUSMSpace
    , details::KokkosSYCLHostUSMSpace
#endif
>;


/// Get the data of an array on CPU, as a `std::vector` in row-major order
/// (taking strides into account)
template <typename T>
static std::vector<T> to_vector(const DataArrayBase& array) {
    // `copy` always creates a contiguous row-major array
    auto copy = array.copy({kDLCPU, 0});
    const auto& host = dynamic_cast<const KokkosDataArray<T, Kokkos::HostSpace>&>(*copy);
    REQUIRE(host.is_contiguous());
    auto size = details::product(host.shape());
    return std::vector<T>(host.data(), host.data() + size);
}

template <typename T, typename MemorySpace>
static std::vector<T> to_vector(const KokkosDataArray<T, MemorySpace>& array) {
    return to_vector<T>(static_cast<const DataArrayBase&>(array));
}

/// Copy `data` to a new view in `MemorySpace`, with the given `layout`
template <typename MemorySpace, typename DataType, typename Layout>
static Kokkos::View<DataType, Layout, MemorySpace> to_space(
    const Kokkos::View<DataType, Layout, Kokkos::HostSpace>& data
) {
    auto view = Kokkos::View<DataType, Layout, MemorySpace>(
        Kokkos::view_alloc("data", Kokkos::WithoutInitializing), data.layout()
    );
    Kokkos::deep_copy(view, data);
    return view;
}

/// Create a `KokkosDataArray` in `MemorySpace` with the given shape and data
template <typename MemorySpace = Kokkos::HostSpace>
static KokkosDataArray<double, MemorySpace> make_array(std::vector<uintptr_t> shape, const std::vector<double>& data) {
    auto host = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>("data", data.size());
    for (size_t i = 0; i < data.size(); i++) {
        host(i) = data[i];
    }
    return KokkosDataArray<double, MemorySpace>(std::move(shape), to_space<MemorySpace>(host));
}

/// Get a vector with values from 0 to `size - 1`
static std::vector<double> vector_range(size_t size) {
    auto values = std::vector<double>(size);
    for (size_t i = 0; i < size; i++) {
        values[i] = static_cast<double>(i);
    }
    return values;
}

/// Create `Labels` with values stored in a `KokkosDataArray` in `MemorySpace`.
/// The labels of a block must be on the same device as the block values.
template <typename MemorySpace>
static Labels make_labels(const std::vector<std::string>& names, const std::vector<std::vector<int32_t>>& values) {
    auto host = Kokkos::View<int32_t*, Kokkos::LayoutRight, Kokkos::HostSpace>("labels", values.size() * names.size());
    for (size_t i = 0; i < values.size(); i++) {
        REQUIRE(values[i].size() == names.size());
        for (size_t j = 0; j < names.size(); j++) {
            host(i * names.size() + j) = values[i][j];
        }
    }

    auto array = std::make_unique<KokkosDataArray<int32_t, MemorySpace>>(
        std::vector<uintptr_t>{values.size(), names.size()},
        to_space<MemorySpace>(host)
    );
    return Labels(names, DataArrayBase::to_mts_array(std::move(array)), assume_unique{});
}

TEST_CASE("KokkosDataArray") {
    SECTION("from n-dimensional views") {
        auto view_2d = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>("view", 2, 3);
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                view_2d(i, j) = static_cast<double>(3 * i + j);
            }
        }

        auto array = KokkosDataArray(view_2d);
        static_assert(std::is_same_v<decltype(array), HostArray>);
        CHECK(array.data() == view_2d.data());

        // the array keeps the data alive
        auto* data_ptr = view_2d.data();
        view_2d = {};
        CHECK(array.data() == data_ptr);
        CHECK(to_vector(array) == std::vector<double>{0, 1, 2, 3, 4, 5});

        // sub-view
        auto view_3x4 = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>("view", 3, 4);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 4; j++) {
                view_3x4(i, j) = static_cast<double>(4 * i + j);
            }
        }
        auto subview = Kokkos::subview(view_3x4, Kokkos::make_pair(1, 3), Kokkos::make_pair(1, 3));
        array = KokkosDataArray(subview);
        CHECK(array.shape() == std::vector<uintptr_t>{2, 2});
        CHECK(array.data() == &view_3x4(1, 1));
        CHECK(array.strides() == std::vector<int64_t>{4, 1});
        CHECK(to_vector(array) == std::vector<double>{5, 6, 9, 10});

        // static extents
        auto view_static = Kokkos::View<double*[2], Kokkos::HostSpace>("view", 3);
        array = KokkosDataArray(view_static);
        CHECK(array.shape() == std::vector<uintptr_t>{3, 2});

        // rank 0
        auto view_scalar = Kokkos::View<double, Kokkos::HostSpace>("scalar");
        view_scalar() = 42.0;
        array = KokkosDataArray(view_scalar);
        CHECK(array.shape().empty());
        CHECK(to_vector(array) == std::vector<double>{42.0});
    }

    SECTION("n-dimensional view accessor") {
        auto array = make_array<Kokkos::HostSpace>({2, 3}, {0, 1, 2, 3, 4, 5});

        auto view = array.view<double**>();
        static_assert(std::is_same_v<decltype(view)::array_layout, Kokkos::LayoutStride>);
        CHECK(view.extent(0) == 2);
        CHECK(view.extent(1) == 3);
        CHECK(view(1, 2) == 5);

        auto right_view = array.view<double**, Kokkos::LayoutRight>();
        CHECK(right_view(1, 2) == 5);

        auto static_view = array.view<double*[3]>();
        CHECK(static_view(1, 0) == 3);

        CHECK_THROWS_WITH(
            (array.view<double**, Kokkos::LayoutLeft>()),
            "this array is not contiguous in column-major order, and can not be "
            "viewed with Kokkos::LayoutLeft in KokkosDataArray::view, use "
            "Kokkos::LayoutStride instead"
        );

        // after swapping the axes, the data is in column-major order
        array.swap_axes(0, 1);
        view = array.view<double**>();
        CHECK(view.extent(0) == 3);
        CHECK(view.extent(1) == 2);
        CHECK(view(2, 1) == 5);
        CHECK(view(0, 1) == 3);

        auto left_view = array.view<double**, Kokkos::LayoutLeft>();
        CHECK(left_view(2, 1) == 5);

        CHECK_THROWS_WITH(
            (array.view<double**, Kokkos::LayoutRight>()),
            "this array is not contiguous in row-major order, and can not be "
            "viewed with Kokkos::LayoutRight in KokkosDataArray::view, use "
            "Kokkos::LayoutStride instead"
        );
        array.swap_axes(0, 1);

        CHECK_THROWS_WITH(
            array.view<double*>(),
            "invalid rank in KokkosDataArray::view: the array has 2 dimensions, "
            "but the requested view has 1"
        );

        CHECK_THROWS_WITH(
            array.view<double*[2]>(),
            "invalid static extent in KokkosDataArray::view: the array has size 3 "
            "for dimension 1, but the requested view has size 2"
        );
    }

    SECTION("errors") {
        auto array = make_array<Kokkos::HostSpace>({2, 3}, {0, 1, 2, 3, 4, 5});

        CHECK_THROWS_WITH(array.reshape({4, 2}), "invalid shape in reshape");
        CHECK_THROWS_WITH(array.swap_axes(0, 3), "out of bounds axis in KokkosDataArray::swap_axes");

        CHECK_THROWS_WITH(
            array.copy({kDLCPU, 1}),
            "requested device (kDLCPU, 1) does not match the device used by Kokkos (kDLCPU, 0)"
        );

#if !defined(KOKKOS_ENABLE_CUDA)
        CHECK_THROWS_WITH(
            array.copy({kDLCUDA, 0}),
            "there is no Kokkos memory space corresponding to DLPack device "
            "(kDLCUDA, 0) in this build of Kokkos"
        );
#endif

        auto fill_value = DataArrayBase::to_mts_array(
            std::make_unique<SimpleDataArray<double>>(std::vector<uintptr_t>{2}, 3.5)
        );
        CHECK_THROWS_WITH(
            array.create({4, 2}, std::move(fill_value)),
            "`fill_value` must be a single scalar"
        );

        int64_t stream = 1;
        CHECK_THROWS_WITH(
            array.as_dlpack({kDLCPU, 0}, &stream, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}),
            "`stream` must be null for CPU data"
        );

        CHECK_THROWS_WITH(
            array.as_dlpack({kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION + 1, 0}),
            "invalid `max_version` in KokkosDataArray::as_dlpack: we got v" +
            std::to_string(DLPACK_MAJOR_VERSION + 1) + ".0, but we support v" +
            std::to_string(DLPACK_MAJOR_VERSION) + "." + std::to_string(DLPACK_MINOR_VERSION)
        );

        CHECK_THROWS_WITH(
            array.from_dlpack(nullptr),
            "dlpack tensor is null in KokkosDataArray::from_dlpack"
        );

        auto* dlpack = array.as_dlpack({kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION});
        dlpack->dl_tensor.dtype = {kDLComplex, 64, 1};
        CHECK_THROWS_WITH(
            array.from_dlpack(dlpack),
            "unsupported DLPack data type: code=5 bits=64"
        );

        auto mts_array = DataArrayBase::to_mts_array(std::make_unique<HostArray>(std::vector<uintptr_t>{2}, 1.0));
        CHECK_THROWS_WITH(
            (KokkosDataArray<float, Kokkos::HostSpace>::from_mts_array(mts_array)),
            "this array is a metatensor::KokkosDataArray, but with a different "
            "data type or memory space"
        );

        auto simple = DataArrayBase::to_mts_array(std::make_unique<SimpleDataArray<double>>(std::vector<uintptr_t>{2}));
        CHECK_THROWS_WITH(
            HostArray::from_mts_array(simple),
            "this array is not a metatensor::KokkosDataArray"
        );
    }

    SECTION("DLPack") {
        auto array = make_array<Kokkos::HostSpace>({2, 3}, {0, 1, 2, 3, 4, 5});
        auto dlpack_array = DLPackArray<double>(array.as_dlpack(
            {kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}
        ));
        CHECK(dlpack_array.shape() == std::vector<size_t>{2, 3});
        CHECK(dlpack_array.data() == array.data());
        CHECK(dlpack_array(1, 2) == 5);

        // the DLPack tensor keeps the data alive
        array = make_array<Kokkos::HostSpace>({1}, {0});
        CHECK(dlpack_array(1, 1) == 4);

        // scalar
        auto scalar = HostArray({}, 3.0);
        auto scalar_dlpack = DLPackArray<double>(scalar.as_dlpack(
            {kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}
        ));
        CHECK(scalar_dlpack.shape().empty());
        CHECK(scalar_dlpack.data()[0] == 3.0);

        // import contiguous data from SimpleDataArray, shared without copy
        auto simple = SimpleDataArray<int64_t>({2, 3}, {0, 1, 2, 3, 4, 5});
        auto imported = array.from_dlpack(simple.as_dlpack(
            {kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}
        ));

        auto& imported_array = dynamic_cast<KokkosDataArray<int64_t, Kokkos::HostSpace>&>(*imported);
        CHECK(imported_array.shape() == std::vector<uintptr_t>{2, 3});
        CHECK(imported_array.data() == simple.data());
        CHECK(to_vector(imported_array) == std::vector<int64_t>{0, 1, 2, 3, 4, 5});
    }
}

TEMPLATE_LIST_TEST_CASE("KokkosDataArray in all memory spaces", "[kokkos]", MemorySpaces) {
    using MemorySpace = TestType;
    using Array = KokkosDataArray<double, MemorySpace>;
    auto dl_device = details::KokkosMemorySpace<MemorySpace>::device();

    SECTION("constructors") {
        auto array = Array({2, 3}, 4.5);
        auto device = array.device();
        CHECK(device.device_type == dl_device.device_type);
        CHECK(device.device_id == dl_device.device_id);

        CHECK(array.shape() == std::vector<uintptr_t>{2, 3});
        CHECK(array.is_contiguous());
        CHECK(to_vector(array) == std::vector<double>(6, 4.5));

        // contiguous view, the data is shared
        auto host_2d = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>("view", 2, 3);
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                host_2d(i, j) = static_cast<double>(3 * i + j);
            }
        }
        auto view_2d = to_space<MemorySpace>(host_2d);

        array = KokkosDataArray(view_2d);
        static_assert(std::is_same_v<decltype(KokkosDataArray(view_2d)), Array>);
        CHECK(array.shape() == std::vector<uintptr_t>{2, 3});
        CHECK(array.data() == view_2d.data());
        CHECK(array.strides() == std::vector<int64_t>{3, 1});
        CHECK(array.is_contiguous());
        CHECK(to_vector(array) == std::vector<double>{0, 1, 2, 3, 4, 5});

        // LayoutLeft view, the data is also shared
        auto host_left = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>("view", 2, 3, 2);
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 2; k++) {
                    host_left(i, j, k) = static_cast<double>(6 * i + 2 * j + k);
                }
            }
        }
        auto view_left = to_space<MemorySpace>(host_left);

        array = KokkosDataArray(view_left);
        CHECK(array.shape() == std::vector<uintptr_t>{2, 3, 2});
        CHECK(array.data() == view_left.data());
        CHECK(array.strides() == std::vector<int64_t>{1, 2, 6});
        CHECK_FALSE(array.is_contiguous());
        CHECK(to_vector(array) == vector_range(12));
    }

    SECTION("reshape") {
        auto array = make_array<MemorySpace>({2, 3}, vector_range(6));
        array.reshape({3, 2});
        CHECK(array.shape() == std::vector<uintptr_t>{3, 2});
        CHECK(to_vector(array) == std::vector<double>{0, 1, 2, 3, 4, 5});

        // reshaping contiguous data does not copy
        array = make_array<MemorySpace>({2, 3, 4}, vector_range(24));
        auto* data_ptr = array.data();
        array.reshape({1, 6, 1, 2, 2, 1});
        CHECK(array.data() == data_ptr);
        CHECK(array.is_contiguous());
        array.reshape({24});
        CHECK(array.data() == data_ptr);
        CHECK(array.strides() == std::vector<int64_t>{1});
        CHECK(to_vector(array) == vector_range(24));

        // reshaping non-contiguous data does not copy when the new shape only
        // splits or merges dimensions which are contiguous with each other
        array = make_array<MemorySpace>({2, 2, 2, 3}, vector_range(24));
        data_ptr = array.data();
        array.swap_axes(0, 1);
        CHECK(array.strides() == std::vector<int64_t>{6, 12, 3, 1});
        // merge the last two dimensions
        array.reshape({2, 2, 6});
        CHECK(array.data() == data_ptr);
        CHECK(array.strides() == std::vector<int64_t>{6, 12, 1});
        // split them again in a different way, and add size-1 dimensions
        array.reshape({2, 1, 2, 3, 2, 1});
        CHECK(array.data() == data_ptr);
        CHECK(array.strides() == std::vector<int64_t>{6, 24, 12, 2, 1, 1});
        CHECK(to_vector(array) == std::vector<double>{
            0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17,
            6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23,
        });

        // reshaping non-contiguous data makes a contiguous copy if the new
        // shape can not be represented with strides
        array = make_array<MemorySpace>({2, 3}, vector_range(6));
        data_ptr = array.data();
        array.swap_axes(0, 1);
        array.reshape({6});
        CHECK(array.shape() == std::vector<uintptr_t>{6});
        CHECK(array.is_contiguous());
        CHECK(array.data() != data_ptr);

        auto device = array.device();
        CHECK(device.device_type == dl_device.device_type);
        CHECK(device.device_id == dl_device.device_id);

        CHECK(to_vector(array) == std::vector<double>{0, 3, 1, 4, 2, 5});
    }

    SECTION("swap_axes") {
        auto array = make_array<MemorySpace>({2, 3}, vector_range(6));
        auto* data_ptr = array.data();
        array.swap_axes(0, 1);
        CHECK(array.shape() == std::vector<uintptr_t>{3, 2});
        // the data is not copied, only the strides change
        CHECK(array.data() == data_ptr);
        CHECK(array.strides() == std::vector<int64_t>{1, 3});
        CHECK(to_vector(array) == std::vector<double>{0, 3, 1, 4, 2, 5});

        array = make_array<MemorySpace>({2, 3, 2}, vector_range(12));
        array.swap_axes(1, 2);
        CHECK(array.shape() == std::vector<uintptr_t>{2, 2, 3});
        CHECK(to_vector(array) == std::vector<double>{0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11});
    }

    SECTION("copy") {
        auto array = make_array<MemorySpace>({2, 3}, vector_range(6));

        // copy to CPU
        auto copy = array.copy({kDLCPU, 0});

        auto device = copy->device();
        CHECK(device.device_type == kDLCPU);
        CHECK(device.device_id == 0);

        CHECK(to_vector<double>(*copy) == vector_range(6));

        // copy on the same device
        copy = array.copy(dl_device);
        device = copy->device();
        CHECK(device.device_type == dl_device.device_type);
        CHECK(device.device_id == dl_device.device_id);

        auto new_array = dynamic_cast<KokkosDataArray<double, MemorySpace>&>(*copy);
        CHECK(new_array.data() != array.data());
        CHECK(to_vector<double>(new_array) == vector_range(6));

        // copying non-contiguous data gives a contiguous array
        array.swap_axes(0, 1);
        copy = array.copy(dl_device);
        new_array = dynamic_cast<KokkosDataArray<double, MemorySpace>&>(*copy);

        CHECK(new_array.shape() == std::vector<uintptr_t>{3, 2});
        CHECK(new_array.strides() == std::vector<int64_t>{2, 1});
        CHECK(to_vector<double>(new_array) == std::vector<double>{0, 3, 1, 4, 2, 5});

        // copy from CPU to this device
        auto host = make_array<Kokkos::HostSpace>({2, 3}, vector_range(6));
        copy = host.copy(dl_device);

        device = copy->device();
        CHECK(device.device_type == dl_device.device_type);
        CHECK(device.device_id == dl_device.device_id);

        CHECK(to_vector<double>(*copy) == vector_range(6));
    }

    SECTION("create") {
        auto array = Array({2, 3}, 1.0);
        auto fill_value = DataArrayBase::to_mts_array(
            std::make_unique<SimpleDataArray<double>>(std::vector<uintptr_t>{}, 3.5)
        );
        auto created = array.create({4, 2}, std::move(fill_value));
        auto& created_array = dynamic_cast<Array&>(*created);
        auto device = created_array.device();

        CHECK(device.device_type == dl_device.device_type);
        CHECK(device.device_id == dl_device.device_id);

        CHECK(created_array.shape() == std::vector<uintptr_t>{4, 2});
        CHECK(to_vector(created_array) == std::vector<double>(8, 3.5));
    }

    SECTION("DLPack export") {
        auto array = make_array<MemorySpace>({2, 3}, vector_range(6));
        array.swap_axes(0, 1);

        // on the same device, the data is shared and the strides are kept
        auto* dlpack = array.as_dlpack(dl_device, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION});
        const auto& tensor = dlpack->dl_tensor;
        CHECK(tensor.device.device_type == dl_device.device_type);
        CHECK(tensor.device.device_id == dl_device.device_id);

        CHECK(tensor.data == static_cast<void*>(array.data()));
        CHECK(std::vector<int64_t>(tensor.shape, tensor.shape + tensor.ndim) == std::vector<int64_t>{3, 2});
        CHECK(std::vector<int64_t>(tensor.strides, tensor.strides + tensor.ndim) == std::vector<int64_t>{1, 3});
        dlpack->deleter(dlpack);

        // on CPU
        auto cpu = DLPackArray<double>(array.as_dlpack(
            {kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}
        ));
        CHECK(cpu.shape() == std::vector<size_t>{3, 2});
        CHECK(cpu(2, 1) == 5);
        CHECK(cpu(0, 1) == 3);

        // through mts_array_t
        auto mts_array = DataArrayBase::to_mts_array(std::make_unique<Array>(std::vector<uintptr_t>{2, 2}, 7.0));
        auto values = mts_array.template as_dlpack_array<double>({kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION});
        CHECK(values.shape() == std::vector<size_t>{2, 2});
        CHECK(values(1, 1) == 7);
    }

    SECTION("DLPack import") {
        auto array = make_array<MemorySpace>({2, 3}, vector_range(6));
        auto* dlpack = array.as_dlpack(dl_device, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION});
        // transpose the data by changing the shape & strides
        std::swap(dlpack->dl_tensor.shape[0], dlpack->dl_tensor.shape[1]);
        std::swap(dlpack->dl_tensor.strides[0], dlpack->dl_tensor.strides[1]);

        // the data is shared, with the same strides
        auto imported = array.from_dlpack(dlpack);
        auto device = imported->device();
        CHECK(device.device_type == dl_device.device_type);
        CHECK(device.device_id == dl_device.device_id);

        auto new_array = dynamic_cast<KokkosDataArray<double, MemorySpace>&>(*imported);
        CHECK(new_array.data() == array.data());
        CHECK(new_array.shape() == std::vector<uintptr_t>{3, 2});
        CHECK(new_array.strides() == std::vector<int64_t>{1, 3});

        CHECK(to_vector<double>(*imported) == std::vector<double>{0, 3, 1, 4, 2, 5});
    }

    SECTION("from_mts_array") {
        auto mts_array = DataArrayBase::to_mts_array(std::make_unique<Array>(std::vector<uintptr_t>{2}, 1.0));
        auto& array = Array::from_mts_array(mts_array);
        CHECK(array.shape() == std::vector<uintptr_t>{2});

        const auto& const_mts_array = mts_array;
        const auto& const_array = Array::from_mts_array(const_mts_array);
        CHECK(&const_array == &array);
    }
}

TEMPLATE_LIST_TEST_CASE("TensorMap with KokkosDataArray", "[kokkos]", MemorySpaces) {
    using MemorySpace = TestType;
    using Array = KokkosDataArray<double, MemorySpace>;

    SECTION("keys_to_properties") {
        auto blocks = std::vector<TensorBlock>();
        blocks.emplace_back(
            std::make_unique<Array>(std::vector<uintptr_t>{2, 1, 2}, 1.0),
            make_labels<MemorySpace>({"samples"}, {{0}, {2}}),
            std::vector<Labels>{make_labels<MemorySpace>({"component"}, {{0}})},
            make_labels<MemorySpace>({"properties"}, {{0}, {1}})
        );
        blocks.emplace_back(
            std::make_unique<Array>(std::vector<uintptr_t>{2, 1, 1}, 2.0),
            make_labels<MemorySpace>({"samples"}, {{1}, {2}}),
            std::vector<Labels>{make_labels<MemorySpace>({"component"}, {{0}})},
            make_labels<MemorySpace>({"properties"}, {{3}})
        );
        auto tensor = TensorMap(make_labels<MemorySpace>({"key"}, {{0}, {1}}), std::move(blocks));

        tensor = tensor.keys_to_properties("key");
        CHECK(tensor.keys().count() == 1);

        auto block = tensor.block_by_id(0);
        CHECK(block.samples() == make_labels<MemorySpace>({"samples"}, {{0}, {1}, {2}}));
        CHECK(block.properties() == make_labels<MemorySpace>({"key", "properties"}, {{0, 0}, {0, 1}, {1, 3}}));

        auto mts_array = block.mts_array();
        const auto& values = Array::from_mts_array(mts_array);
        CHECK(values.shape() == std::vector<uintptr_t>{3, 1, 3});
        CHECK(to_vector(values) == std::vector<double>{
            1.0, 1.0, 0.0,
            0.0, 0.0, 2.0,
            1.0, 1.0, 2.0,
        });
    }

    SECTION("keys_to_samples") {
        auto blocks = std::vector<TensorBlock>();
        blocks.emplace_back(
            std::make_unique<Array>(std::vector<uintptr_t>{2, 2}, 1.0),
            make_labels<MemorySpace>({"samples"}, {{0}, {2}}),
            std::vector<Labels>{},
            make_labels<MemorySpace>({"properties"}, {{0}, {1}})
        );
        blocks.emplace_back(
            std::make_unique<Array>(std::vector<uintptr_t>{2, 2}, 2.0),
            make_labels<MemorySpace>({"samples"}, {{1}, {2}}),
            std::vector<Labels>{},
            make_labels<MemorySpace>({"properties"}, {{0}, {1}})
        );
        auto tensor = TensorMap(make_labels<MemorySpace>({"key"}, {{0}, {1}}), std::move(blocks));

        tensor = tensor.keys_to_samples("key");
        auto block = tensor.block_by_id(0);
        CHECK(block.samples() == make_labels<MemorySpace>({"samples", "key"}, {{0, 0}, {1, 1}, {2, 0}, {2, 1}}));

        auto mts_array = block.mts_array();
        const auto& values = Array::from_mts_array(mts_array);
        CHECK(values.shape() == std::vector<uintptr_t>{4, 2});
        CHECK(to_vector(values) == std::vector<double>{1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0});
    }

    SECTION("non-contiguous data") {
        // values[s, c, p] = 100 * s + 10 * c + p, stored in column-major order
        auto values_1 = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>("values", 2, 2, 3);
        auto values_2 = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>("values", 1, 2, 3);
        for (size_t c = 0; c < 2; c++) {
            for (size_t p = 0; p < 3; p++) {
                for (size_t s = 0; s < 2; s++) {
                    values_1(s, c, p) = static_cast<double>(100 * s + 10 * c + p);
                }
                values_2(0, c, p) = static_cast<double>(1000 + 10 * c + p);
            }
        }

        auto blocks = std::vector<TensorBlock>();
        blocks.emplace_back(
            std::make_unique<Array>(to_space<MemorySpace>(values_1)),
            make_labels<MemorySpace>({"samples"}, {{0}, {1}}),
            std::vector<Labels>{make_labels<MemorySpace>({"component"}, {{0}, {1}})},
            make_labels<MemorySpace>({"properties"}, {{0}, {1}, {2}})
        );
        blocks.emplace_back(
            std::make_unique<Array>(to_space<MemorySpace>(values_2)),
            make_labels<MemorySpace>({"samples"}, {{1}}),
            std::vector<Labels>{make_labels<MemorySpace>({"component"}, {{0}, {1}})},
            make_labels<MemorySpace>({"properties"}, {{0}, {1}, {2}})
        );
        auto tensor = TensorMap(make_labels<MemorySpace>({"key"}, {{0}, {1}}), std::move(blocks));

        // move_data reading from strided input arrays
        auto merged = tensor.keys_to_samples("key");
        auto block = merged.block_by_id(0);
        CHECK(block.samples() == make_labels<MemorySpace>({"samples", "key"}, {{0, 0}, {1, 0}, {1, 1}}));

        auto mts_array = block.mts_array();
        const auto& values = Array::from_mts_array(mts_array);
        CHECK(to_vector(values) == std::vector<double>{
              0,   1,   2,   10,   11,   12,
            100, 101, 102,  110,  111,  112,
           1000, 1001, 1002, 1010, 1011, 1012,
        });

        // serialization uses as_dlpack, which exports the strides on CPU
        auto buffer = io::save_buffer(tensor);
        auto loaded = TensorMap::load_buffer(buffer);
        auto loaded_values = loaded.block_by_id(0).values();
        CHECK(loaded_values.shape() == std::vector<size_t>{2, 2, 3});
        CHECK(loaded_values(1, 0, 2) == 102);
        CHECK(loaded_values(0, 1, 1) == 11);
    }
}
