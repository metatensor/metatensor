#include <cstring>
#include <memory>

#include <catch.hpp>

#include <metatensor.hpp>
using namespace metatensor;

TEST_CASE("Array sizes") {
    // Any change here means the ABI will break and the new version will need to
    // be a major version bump.
    CHECK(sizeof(MtsArray) == 128);
    CHECK(sizeof(mts_array_t) == 104);

    CHECK(sizeof(DataArrayBase) == 8);

    size_t base_size = sizeof(void*)
                    + sizeof(std::vector<size_t>)
                    + sizeof(std::vector<int64_t>)
                    + sizeof(bool) + 7 // padding
                    + sizeof(void*)
                    + sizeof(std::function<void(void*)>);
    CHECK(sizeof(NDArray<double>) == base_size + 16);
    CHECK(sizeof(NDArray<char>) == base_size + 16);

    base_size = sizeof(DataArrayBase)
              + sizeof(std::vector<uintptr_t>)
              + sizeof(std::vector<int64_t>)
              + sizeof(std::shared_ptr<std::vector<double>>);
    CHECK(sizeof(SimpleDataArray<double>) == base_size + 16);
    CHECK(sizeof(SimpleDataArray<char>) == base_size + 16);

    base_size = 2 * sizeof(void*)
              + sizeof(std::vector<size_t>)
              + sizeof(std::vector<int64_t>);
    CHECK(sizeof(DLPackArray<double>) == base_size + 16);
    CHECK(sizeof(DLPackArray<char>) == base_size + 16);
}

TEST_CASE("Data Array") {
    auto data = std::make_unique<SimpleDataArray<double>>(SimpleDataArray<double>({2, 3, 4}));
    auto array = DataArrayBase::to_mts_array(std::move(data));

    SECTION("origin") {
        auto origin = array.origin();

        char buffer[64] = {0};
        auto status = mts_get_data_origin(origin, buffer, 64);
        CHECK(status == MTS_SUCCESS);
        CHECK(std::string(buffer) == "metatensor::SimpleDataArray");
    }

    SECTION("shape") {
        CHECK(array.shape() == std::vector<uintptr_t>{2, 3, 4});
        auto new_shape = std::vector<uintptr_t>{1, 2, 3, 4};

        array.reshape(new_shape);
        CHECK(array.shape() == new_shape);

        array.swap_axes(1, 2);
        CHECK(array.shape() == std::vector<uintptr_t>{1, 3, 2, 4});
    }

    SECTION("new arrays") {
        MtsArray new_array = array;
        CHECK(new_array.as_mts_array_t().ptr != array.as_mts_array_t().ptr);
        CHECK(new_array.shape() == std::vector<uintptr_t>{2, 3, 4});


        auto new_shape = std::vector<uintptr_t>{1, 2, 3, 4};
        auto fill_value = DataArrayBase::to_mts_array(
            std::make_unique<SimpleDataArray<double>>(std::vector<uintptr_t>{}, 0.0)
        );
        new_array = array.create(new_shape, std::move(fill_value));
        CHECK(new_array.shape() == new_shape);
    }
}

TEST_CASE("SimpleDataArray - reshape and swap_axes") {
    auto array = SimpleDataArray<double>({2, 3}, {0.0, 1.1, 2.2, 3.3, 4.4, 5.5});

    array.swap_axes(0, 1);
    CHECK(array.shape() == std::vector<uintptr_t>{3, 2});
    CHECK(array.strides() == std::vector<int64_t>{2, 1});
    CHECK(array == SimpleDataArray<double>({3, 2}, {0.0, 3.3, 1.1, 4.4, 2.2, 5.5}));

    array = SimpleDataArray<double>({2, 3, 2}, {0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11});
    array.swap_axes(1, 2);
    CHECK(array.shape() == std::vector<uintptr_t>{2, 2, 3});
    CHECK(array.strides() == std::vector<int64_t>{6, 3, 1});
    CHECK(array == SimpleDataArray<double>({2, 2, 3}, {0.0, 2.2, 4.4, 1.1, 3.3, 5.5, 6.6, 8.8, 10.10, 7.7, 9.9, 11.11}));

    array.swap_axes(0, 2);
    CHECK(array.shape() == std::vector<uintptr_t>{3, 2, 2});
    CHECK(array == SimpleDataArray<double>({3, 2, 2}, {0.0, 6.6, 1.1, 7.7, 2.2, 8.8, 3.3, 9.9, 4.4, 10.10, 5.5, 11.11}));

    CHECK_THROWS_WITH(array.swap_axes(0, 3), "out of bounds axis in swap_axes");

    array.reshape({4, 3});
    CHECK(array.shape() == std::vector<uintptr_t>{4, 3});
    CHECK(array.strides() == std::vector<int64_t>{3, 1});
    CHECK(array == SimpleDataArray<double>({4, 3}, {0.0, 6.6, 1.1, 7.7, 2.2, 8.8, 3.3, 9.9, 4.4, 10.10, 5.5, 11.11}));
}

TEMPLATE_TEST_CASE("SimpleDataArray - all dtypes", "",
    double, float, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bool
) {
    using storage_t = details::storage_type_t<TestType>;
    auto cpu_device = DLDevice{kDLCPU, 0};
    auto version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};

    SECTION("as_dlpack") {
        auto data = std::make_unique<SimpleDataArray<TestType>>(std::vector<uintptr_t>{2, 3, 4});
        {
            auto view = data->view();
            view(1, 1, 0) = 1;
        }
        MtsArray array = DataArrayBase::to_mts_array(std::move(data));

        auto dlpack_array = array.as_dlpack_array<TestType>(cpu_device, nullptr, version);
        CHECK(dlpack_array.device().device_type == kDLCPU);
        CHECK(dlpack_array.device().device_id == 0);
        CHECK(details::dlpack_dtype_matches<TestType>(dlpack_array.dtype()));
        CHECK(dlpack_array.shape() == std::vector<uintptr_t>{2, 3, 4});
        CHECK(dlpack_array.data()[16] == static_cast<TestType>(1));
    }

    SECTION("default_create_array") {
        auto shape = std::vector<uintptr_t>{2, 3};
        mts_array_t raw_array = {};
        details::check_status(details::default_create_array(
            shape.data(), shape.size(), details::dtype_of<TestType>(), &raw_array
        ));
        auto array = MtsArray(raw_array);

        CHECK(details::dlpack_dtype_matches<TestType>(array.dtype()));
        CHECK(SimpleDataArray<TestType>::from_mts_array(array) == SimpleDataArray<TestType>({2, 3}));
    }

    SECTION("from_dlpack") {
        auto data = SimpleDataArray<TestType>({2, 2}, std::vector<storage_t>{0, 1, 1, 0});

        // from_dlpack can create arrays with a different dtype than the
        // array it is called on
        auto source = DataArrayBase::to_mts_array(
            std::make_unique<SimpleDataArray<double>>(std::vector<uintptr_t>{1})
        );
        auto array = source.from_dlpack(data.as_dlpack(cpu_device, nullptr, version));

        CHECK(details::dlpack_dtype_matches<TestType>(array.dtype()));
        CHECK(SimpleDataArray<TestType>::from_mts_array(array) == data);
    }

    SECTION("create") {
        auto array = DataArrayBase::to_mts_array(
            std::make_unique<SimpleDataArray<TestType>>(std::vector<uintptr_t>{2, 2})
        );
        auto fill_value = DataArrayBase::to_mts_array(
            std::make_unique<SimpleDataArray<TestType>>(std::vector<uintptr_t>{}, static_cast<TestType>(1))
        );

        auto new_array = array.create({3, 2}, std::move(fill_value));
        CHECK(details::dlpack_dtype_matches<TestType>(new_array.dtype()));
        CHECK(
            SimpleDataArray<TestType>::from_mts_array(new_array)
            == SimpleDataArray<TestType>({3, 2}, static_cast<TestType>(1))
        );
    }

    SECTION("create_empty_array") {
        auto array = details::create_empty_array({2, 3}, details::dtype_of<TestType>());
        REQUIRE(dynamic_cast<EmptyDataArray<TestType>*>(array.get()) != nullptr);
        CHECK(array->shape() == std::vector<uintptr_t>{2, 3});
        CHECK(details::dlpack_dtype_matches<TestType>(array->dtype()));
    }
}

TEST_CASE("DLPackArray<T> - construction and access") {
    // Create a SimpleDataArray<double>, get a DLPack tensor, wrap in DLPackArray
    auto data = SimpleDataArray<double>({2, 3});
    auto view = data.view();
    view(0, 0) = 1.0;
    view(0, 1) = 2.0;
    view(0, 2) = 3.0;
    view(1, 0) = 4.0;
    view(1, 1) = 5.0;
    view(1, 2) = 6.0;

    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    DLManagedTensorVersioned* managed = data.as_dlpack(cpu_device, nullptr, ver);
    REQUIRE(managed != nullptr);

    SECTION("basic construction and element access") {
        auto arr = DLPackArray<double>(managed);
        CHECK(arr.shape().size() == 2);
        CHECK(arr.shape()[0] == 2);
        CHECK(arr.shape()[1] == 3);

        CHECK(arr(0, 0) == 1.0);
        CHECK(arr(0, 1) == 2.0);
        CHECK(arr(0, 2) == 3.0);
        CHECK(arr(1, 0) == 4.0);
        CHECK(arr(1, 1) == 5.0);
        CHECK(arr(1, 2) == 6.0);

        CHECK(arr.data() != nullptr);
        CHECK(arr.data()[0] == 1.0);
        // DLPackArray destructor will call the deleter
    }
}

TEST_CASE("DLPackArray<T> - dtype mismatch") {
    auto data = SimpleDataArray<double>({2, 3});

    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    DLManagedTensorVersioned* managed = data.as_dlpack(cpu_device, nullptr, ver);
    REQUIRE(managed != nullptr);

    // double tensor should NOT match int32_t
    CHECK_THROWS_WITH(
        DLPackArray<int32_t>(managed),
        Catch::Matchers::Contains("dtype mismatch")
    );

    // After dtype mismatch, the managed tensor should have been cleaned up.
    // Get a new one for float mismatch test
    managed = data.as_dlpack(cpu_device, nullptr, ver);
    REQUIRE(managed != nullptr);

    CHECK_THROWS_WITH(
        DLPackArray<float>(managed),
        Catch::Matchers::Contains("dtype mismatch")
    );
}

TEST_CASE("DLPackArray<T> - with int32 data") {
    auto data = SimpleDataArray<int32_t>({2, 2});
    auto view = data.view();
    view(0, 0) = 10;
    view(0, 1) = 20;
    view(1, 0) = 30;
    view(1, 1) = 40;

    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    DLManagedTensorVersioned* managed = data.as_dlpack(cpu_device, nullptr, ver);
    REQUIRE(managed != nullptr);

    auto arr = DLPackArray<int32_t>(managed);
    CHECK(arr(0, 0) == 10);
    CHECK(arr(0, 1) == 20);
    CHECK(arr(1, 0) == 30);
    CHECK(arr(1, 1) == 40);
}

TEST_CASE("DLPackArray<T> - empty array") {
    auto array = DLPackArray<double>();
    CHECK(array.data() == nullptr);
    CHECK(array.shape() == std::vector<size_t>{});

    array = DLPackArray<double>(nullptr);
    CHECK(array.data() == nullptr);
    CHECK(array.shape() == std::vector<size_t>{});
}

TEST_CASE("SimpleDataArray - device()") {
    auto data = std::make_unique<SimpleDataArray<double>>(SimpleDataArray<double>({2, 3}));

    // direct call to device()
    CHECK(data->device().device_type == kDLCPU);
    CHECK(data->device().device_id == 0);

    // through mts_array_t callback
    auto array = DataArrayBase::to_mts_array(std::move(data));
    CHECK(array.device().device_type == kDLCPU);
    CHECK(array.device().device_id == 0);
}

TEST_CASE("EmptyDataArray - device()") {
    auto data = EmptyDataArray<double>({2, 3});
    auto dev = data.device();
    CHECK(dev.device_type == kDLCPU);
    CHECK(dev.device_id == 0);
}

TEST_CASE("DLPackArray<T> - device()") {
    SECTION("empty array returns CPU") {
        auto arr = DLPackArray<double>();
        auto dev = arr.device();
        CHECK(dev.device_type == kDLCPU);
        CHECK(dev.device_id == 0);
    }

    SECTION("CPU array returns CPU") {
        auto data = SimpleDataArray<double>({2, 3});
        DLDevice cpu_device = {kDLCPU, 0};
        DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        DLManagedTensorVersioned* managed = data.as_dlpack(cpu_device, nullptr, ver);
        REQUIRE(managed != nullptr);

        auto arr = DLPackArray<double>(managed);
        auto dev = arr.device();
        CHECK(dev.device_type == kDLCPU);
        CHECK(dev.device_id == 0);
    }
}

TEST_CASE("TensorBlock::values() with device parameter") {
    auto data = std::make_unique<SimpleDataArray<double>>(SimpleDataArray<double>({2, 3}, 1.0));
    auto samples = Labels({"s"}, {{0}, {1}});
    auto properties = Labels({"p"}, {{0}, {1}, {2}});

    auto block = TensorBlock(
        std::move(data),
        samples,
        {},
        properties
    );

    SECTION("default (CPU)") {
        auto values = block.values();
        CHECK(values.shape().size() == 2);
        CHECK(values.shape()[0] == 2);
        CHECK(values.shape()[1] == 3);
        CHECK(values(0, 0) == 1.0);
    }

    SECTION("explicit CPU device") {
        DLDevice cpu = {kDLCPU, 0};
        auto values = block.values(cpu);
        CHECK(values.shape()[0] == 2);
        CHECK(values(0, 0) == 1.0);
    }
}

TEST_CASE("SimpleDataArray - DLPack version mismatch") {
    auto data = std::make_unique<SimpleDataArray<double>>(SimpleDataArray<double>({2, 2}));

    DLDevice cpu_device = {kDLCPU, 0};

    // Case 1: Request an older, incompatible version (0.1)
    // The implementation supports 1.1, so this must fail.
    DLPackVersion old_version = {0, 1};
    CHECK_THROWS_WITH(
        data->as_dlpack(cpu_device, nullptr, old_version),
        "invalid `max_version` in SimpleDataArray::as_dlpack: "
        "we got v0.1, but we support v1.3"
    );

    // Case 2: Request a newer version (1.5)
    // Succeed because 1.5 is compatible with 1.3
    DLPackVersion new_version = {1, 5};
    DLManagedTensorVersioned* managed = nullptr;
    managed = data->as_dlpack(cpu_device, nullptr, new_version);

    // Case 3: Request an ABI breaking version (2.0)
    // Succeed because 1.5 is compatible with 1.0
    DLPackVersion too_new_version = {2, 0};
    CHECK_THROWS_WITH(
        data->as_dlpack(cpu_device, nullptr, too_new_version),
        "invalid `max_version` in SimpleDataArray::as_dlpack: "
        "we got v2.0, but we support v1.3"
    );

    // Verify we got back the implementation version, not the requested one
    REQUIRE(managed != nullptr);
    CHECK(managed->version.major == DLPACK_MAJOR_VERSION);
    CHECK(managed->version.minor <= 3);

    managed->deleter(managed);
}


template<typename T>
void check_array_equality() {
    auto array_1 = SimpleDataArray<T>({2, 2}, {1, 2, 3, 4});
    auto array_2 = SimpleDataArray<T>({2, 2}, {1, 2, 3, -4});
    auto array_3 = SimpleDataArray<T>({2, 2}, {1, 2, 3, 4});

    // SimpleDataArray equality
    CHECK(array_1 == array_3);
    CHECK(array_1 != array_2);

    // NDArray equality
    CHECK(array_1.view() == array_3.view());
    CHECK(array_1.view() != array_2.view());

    auto cpu_device = DLDevice{kDLCPU, 0};
    auto ver = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto dlpack_1 = DLPackArray<T>(array_1.as_dlpack(cpu_device, nullptr, ver));
    auto dlpack_2 = DLPackArray<T>(array_2.as_dlpack(cpu_device, nullptr, ver));
    auto dlpack_3 = DLPackArray<T>(array_3.as_dlpack(cpu_device, nullptr, ver));

    // DLPackArray equality
    CHECK(dlpack_1 == dlpack_3);
    CHECK(dlpack_1 != dlpack_2);

    // SimpleDataArray and NDArray
    CHECK(array_1 == array_3.view());
    CHECK(array_3.view() == array_1);
    CHECK(array_1 != array_2.view());
    CHECK(array_2.view() != array_1);

    // SimpleDataArray and DLPackArray
    CHECK(dlpack_1 == array_3);
    CHECK(array_3 == dlpack_1);
    CHECK(dlpack_1 != array_2);
    CHECK(array_2 != dlpack_1);

    // NDArray and DLPackArray
    CHECK(dlpack_1 == array_3.view());
    CHECK(array_3.view() == dlpack_1);
    CHECK(dlpack_1 != array_2.view());
    CHECK(array_2.view() != dlpack_1);
}

TEST_CASE("array equality") {
    check_array_equality<double>();
    check_array_equality<int>();
}

struct NonContiguousInt32DLPackContext {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::shared_ptr<std::vector<int32_t>> data;
};

static void non_contiguous_int32_dlpack_deleter(DLManagedTensorVersioned* self) {
    if (self != nullptr) {
        delete static_cast<NonContiguousInt32DLPackContext*>(self->manager_ctx);
        delete self;
    }
}

static DLManagedTensorVersioned* make_non_contiguous_int32_dlpack(
    std::shared_ptr<std::vector<int32_t>> data
) {
    auto* managed = new DLManagedTensorVersioned();
    auto* context = new NonContiguousInt32DLPackContext();

    context->shape = {3, 2};
    // Expose a transposed non-contiguous view (torch-like stride)
    context->strides = {1, 3};
    context->data = std::move(data);

    managed->version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    managed->manager_ctx = context;
    managed->deleter = non_contiguous_int32_dlpack_deleter;
    managed->flags = 0;

    managed->dl_tensor.data = context->data->data();
    managed->dl_tensor.device = {kDLCPU, 0};
    managed->dl_tensor.ndim = 2;
    managed->dl_tensor.dtype = metatensor::details::dtype_of<int32_t>();
    managed->dl_tensor.shape = context->shape.data();
    managed->dl_tensor.strides = context->strides.data();
    managed->dl_tensor.byte_offset = 0;

    return managed;
}

TEST_CASE("Array equality with non-contiguous strides") {
    auto non_contiguous_data = std::make_shared<std::vector<int32_t>>(
        std::vector<int32_t>{1, 2, 3, 4, 3, 1}
    );
    auto contiguous = SimpleDataArray<int32_t>({3, 2}, {1, 4, 2, 3, 3, 1});

    auto version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};

    auto non_contiguous_dlpack = DLPackArray<int32_t>(
        make_non_contiguous_int32_dlpack(std::move(non_contiguous_data))
    );
    auto contiguous_dlpack = DLPackArray<int32_t>(
        contiguous.as_dlpack({kDLCPU, 0}, nullptr, version)
    );

    CHECK(non_contiguous_dlpack.shape() == std::vector<size_t>{3, 2});
    CHECK(contiguous_dlpack.shape() == std::vector<size_t>{3, 2});
    CHECK(non_contiguous_dlpack == contiguous_dlpack);
    CHECK(contiguous_dlpack == non_contiguous_dlpack);
}


struct BoolDLPackContext {
    std::vector<int64_t> shape;
    std::shared_ptr<std::vector<uint8_t>> data;
};

static void bool_dlpack_deleter(DLManagedTensorVersioned* self) {
    if (self != nullptr) {
        delete static_cast<BoolDLPackContext*>(self->manager_ctx);
        delete self;
    }
}

/// Build a contiguous DLPack tensor with `kDLBool` dtype from a vector of
/// 0/1 bytes. The caller owns the returned managed tensor.
static DLManagedTensorVersioned* make_bool_dlpack(
    std::shared_ptr<std::vector<uint8_t>> data,
    std::vector<int64_t> shape
) {
    auto* managed = new DLManagedTensorVersioned();
    auto* context = new BoolDLPackContext();

    context->shape = std::move(shape);
    context->data = std::move(data);

    managed->version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    managed->manager_ctx = context;
    managed->deleter = bool_dlpack_deleter;
    managed->flags = 0;

    managed->dl_tensor.data = context->data->data();
    managed->dl_tensor.device = {kDLCPU, 0};
    managed->dl_tensor.ndim = static_cast<int32_t>(context->shape.size());
    managed->dl_tensor.dtype = {kDLBool, 8, 1};
    managed->dl_tensor.shape = context->shape.data();
    managed->dl_tensor.strides = nullptr;
    managed->dl_tensor.byte_offset = 0;

    return managed;
}


TEST_CASE("DLPackArray<bool>") {
    SECTION("construction and access") {
        auto raw_data = std::make_shared<std::vector<uint8_t>>(
            std::vector<uint8_t>{1, 0, 1, 1}
        );
        auto* dl_managed = make_bool_dlpack(raw_data, {2, 2});

        auto arr = DLPackArray<bool>(dl_managed);

        CHECK(arr.shape() == std::vector<size_t>{2, 2});

        auto dt = arr.dtype();
        CHECK(dt.code == kDLBool);
        CHECK(dt.bits == 8);
        CHECK(dt.lanes == 1);

        CHECK(arr(0, 0) == true);
        CHECK(arr(0, 1) == false);
        CHECK(arr(1, 0) == true);
        CHECK(arr(1, 1) == true);

        CHECK(arr.data()[0] == true);
        CHECK(arr.data()[1] == false);
        CHECK(arr.data()[2] == true);
        CHECK(arr.data()[3] == true);
    }

    SECTION("dtype mismatch with uint8") {
        // A kDLBool DLPack tensor must NOT be accepted by DLPackArray<uint8_t>
        auto raw_data = std::make_shared<std::vector<uint8_t>>(
            std::vector<uint8_t>{1, 0, 1, 1}
        );
        auto* bool_managed = make_bool_dlpack(raw_data, {2, 2});

        CHECK_THROWS_WITH(
            DLPackArray<uint8_t>(bool_managed),
            Catch::Matchers::Contains("dtype mismatch")
        );

        // A kDLUInt DLPack tensor must NOT be accepted by DLPackArray<bool>
        auto uint8_data = SimpleDataArray<uint8_t>({2, 2});
        auto version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        auto* uint8_managed = uint8_data.as_dlpack({kDLCPU, 0}, nullptr, version);

        CHECK_THROWS_WITH(
            DLPackArray<bool>(uint8_managed),
            Catch::Matchers::Contains("dtype mismatch")
        );
    }
}

TEST_CASE("Unsupported dtypes") {
    auto shape = std::vector<uintptr_t>{2, 3};
    auto complex = DLDataType{kDLComplex, 128, 1};
    auto vector = DLDataType{kDLFloat, 64, 2};

    mts_array_t raw_array = {};
    CHECK_THROWS_WITH(
        details::check_status(details::default_create_array(shape.data(), shape.size(), complex, &raw_array)),
        "unsupported DLDataType in default_create_array: code=5 bits=128"
    );
    CHECK_THROWS_WITH(
        details::check_status(details::default_create_array(shape.data(), shape.size(), vector, &raw_array)),
        "unsupported DLDataType in default_create_array: lanes=2 (expected 1)"
    );

    // reset the last error, this should free the exception from
    // `default_create_array` and avoid memory leak
    auto status = mts_set_last_error(nullptr, nullptr, nullptr, nullptr);
    CHECK(status == MTS_SUCCESS);

    CHECK_THROWS_WITH(
        details::create_empty_array(shape, complex),
        "unsupported DLDataType in create_empty_array: code=5 bits=128"
    );
    CHECK_THROWS_WITH(
        details::create_empty_array(shape, vector),
        "unsupported DLDataType in create_empty_array: lanes=2 (expected 1)"
    );
}

TEST_CASE("SimpleDataArray - errors") {
    auto cpu_device = DLDevice{kDLCPU, 0};
    auto cuda_device = DLDevice{kDLCUDA, 0};
    auto version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};

    CHECK_THROWS_WITH(
        SimpleDataArray<double>({2, 3}, std::vector<double>{1, 2, 3}),
        "the shape and size of the data don't match in SimpleDataArray"
    );

    auto array = SimpleDataArray<double>({2, 3});
    CHECK_THROWS_WITH(array.reshape({4, 2}), "invalid shape in reshape");
    CHECK_THROWS_WITH(array.copy(cuda_device), "SimpleDataArray only supports copying to CPU");

    CHECK_THROWS_WITH(
        array.as_dlpack(cuda_device, nullptr, version),
        "SimpleDataArray only supports CPU device (kDLCPU)"
    );
    int64_t stream = 0;
    CHECK_THROWS_WITH(
        array.as_dlpack(cpu_device, &stream, version),
        "`stream` must be null for CPU data"
    );

    auto fill_value = DataArrayBase::to_mts_array(
        std::make_unique<SimpleDataArray<double>>(std::vector<uintptr_t>{1}, 1.0)
    );
    CHECK_THROWS_WITH(
        array.create({2, 2}, std::move(fill_value)),
        "`fill_value` must be a single scalar"
    );

    // moving data with an empty list of moves does nothing
    auto other = SimpleDataArray<double>({2, 3}, 1.0);
    array.move_data(other, {});
    CHECK(array == SimpleDataArray<double>({2, 3}));

    CHECK_THROWS_WITH(
        array.from_dlpack(nullptr),
        "dlpack tensor is null in SimpleDataArray::from_dlpack"
    );

    auto* managed = array.as_dlpack(cpu_device, nullptr, version);
    managed->version = {0, 8};
    CHECK_THROWS_WITH(
        array.from_dlpack(managed),
        "DLPack version mismatch in SimpleDataArray::from_dlpack: expected major 1, got 0.8"
    );

    managed = array.as_dlpack(cpu_device, nullptr, version);
    managed->dl_tensor.device = cuda_device;
    CHECK_THROWS_WITH(
        array.from_dlpack(managed),
        "SimpleDataArray::from_dlpack only supports CPU device (kDLCPU), got device type 2"
    );

    managed = array.as_dlpack(cpu_device, nullptr, version);
    managed->dl_tensor.dtype.lanes = 2;
    CHECK_THROWS_WITH(
        array.from_dlpack(managed),
        "SimpleDataArray::from_dlpack only supports DLPack tensors with lanes == 1"
    );

    managed = array.as_dlpack(cpu_device, nullptr, version);
    managed->dl_tensor.dtype = DLDataType{kDLComplex, 128, 1};
    CHECK_THROWS_WITH(
        array.from_dlpack(managed),
        "unsupported DLDataType in SimpleDataArray::from_dlpack: code=5 bits=128"
    );
}

TEST_CASE("EmptyDataArray") {
    auto array = EmptyDataArray<float>({2, 3, 4});

    array.reshape({6, 4});
    CHECK(array.shape() == std::vector<uintptr_t>{6, 4});
    CHECK_THROWS_WITH(array.reshape({5, 4}), "invalid shape in reshape");

    array.swap_axes(0, 1);
    CHECK(array.shape() == std::vector<uintptr_t>{4, 6});
    CHECK_THROWS_WITH(array.swap_axes(0, 2), "out of bounds axis in swap_axes");

    auto copy = array.copy(DLDevice{kDLCPU, 0});
    REQUIRE(dynamic_cast<EmptyDataArray<float>*>(copy.get()) != nullptr);
    CHECK(copy->shape() == std::vector<uintptr_t>{4, 6});
    CHECK_THROWS_WITH(array.copy(DLDevice{kDLCUDA, 0}), "EmptyDataArray only supports copying to CPU");

    auto created = array.create({3, 2}, MtsArray(mts_array_t{}));
    REQUIRE(dynamic_cast<EmptyDataArray<float>*>(created.get()) != nullptr);
    CHECK(created->shape() == std::vector<uintptr_t>{3, 2});

    auto version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    CHECK_THROWS_WITH(
        array.as_dlpack(DLDevice{kDLCPU, 0}, nullptr, version),
        "can not call `as_dlpack` for an EmptyDataArray"
    );
    CHECK_THROWS_WITH(
        array.from_dlpack(nullptr),
        "can not call `from_dlpack` for an EmptyDataArray"
    );
    CHECK_THROWS_WITH(
        array.move_data(*copy, {}),
        "can not call `move_data` for an EmptyDataArray"
    );
}
