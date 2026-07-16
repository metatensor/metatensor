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

TEST_CASE("SimpleDataArray<double> - as_dlpack()") {
    auto data = std::make_unique<SimpleDataArray<double>>(SimpleDataArray<double>({2, 3, 4}));
    {
        auto view = data->view();
        view(1, 1, 0) = 1.2345;
    }
    auto array = DataArrayBase::to_mts_array(std::move(data));

    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto dlpack_array = array.as_dlpack_array<double>(cpu_device, nullptr, ver);

    CHECK(dlpack_array.device().device_type == kDLCPU);
    CHECK(dlpack_array.device().device_id == 0);

    CHECK(dlpack_array.dtype().code == kDLFloat);
    CHECK(dlpack_array.dtype().bits == 64);
    CHECK(dlpack_array.dtype().lanes == 1);

    CHECK(dlpack_array.shape() == std::vector<uintptr_t>{2, 3, 4});

    const auto* data_ptr = dlpack_array.data();
    CHECK(data_ptr[16] == 1.2345);
}

TEST_CASE("SimpleDataArray<float> - as_dlpack()") {
    // Create float-backed array via C++ class, expose through mts_array_t
    auto data = std::make_unique<SimpleDataArray<float>>(SimpleDataArray<float>({2, 3, 4}));
    {
        auto view = data->view();
        view(1, 1, 0) = 3.1415F;
    }

    auto array = DataArrayBase::to_mts_array(std::move(data));

    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto dlpack_array = array.as_dlpack_array<float>(cpu_device, nullptr, ver);

    CHECK(dlpack_array.device().device_type == kDLCPU);
    CHECK(dlpack_array.device().device_id == 0);

    CHECK(dlpack_array.dtype().code == kDLFloat);
    CHECK(dlpack_array.dtype().bits == 32);
    CHECK(dlpack_array.dtype().lanes == 1);

    CHECK(dlpack_array.shape() == std::vector<uintptr_t>{2, 3, 4});

    const auto* data_ptr = dlpack_array.data();
    CHECK(data_ptr[16] == 3.1415F);
}

TEST_CASE("SimpleDataArray<int32_t> - as_dlpack()") {
    auto data = std::make_unique<SimpleDataArray<int32_t>>(SimpleDataArray<int32_t>({2, 3, 4}));
    {
        auto view = data->view();
        view(1, 1, 0) = 42;
    }

    auto array = DataArrayBase::to_mts_array(std::move(data));

    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto dlpack_array = array.as_dlpack_array<int32_t>(cpu_device, nullptr, ver);

    CHECK(dlpack_array.device().device_type == kDLCPU);
    CHECK(dlpack_array.device().device_id == 0);

    CHECK(dlpack_array.dtype().code == kDLInt);
    CHECK(dlpack_array.dtype().bits == 32);
    CHECK(dlpack_array.dtype().lanes == 1);

    CHECK(dlpack_array.shape() == std::vector<uintptr_t>{2, 3, 4});

    const auto* data_ptr = dlpack_array.data();
    CHECK(data_ptr[16] == 42);
}

TEST_CASE("SimpleDataArray - from_dlpack()") {
    auto double_data = std::make_unique<SimpleDataArray<double>>(SimpleDataArray<double>({2, 3, 4}));
    auto double_array = DataArrayBase::to_mts_array(std::move(double_data));

    auto int_data = std::make_unique<SimpleDataArray<int16_t>>(SimpleDataArray<int16_t>({2, 3, 4}));
    auto int_array = DataArrayBase::to_mts_array(std::move(int_data));

    auto* double_dl_tensor = double_array.as_dlpack({kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION});
    auto* int_dl_tensor = int_array.as_dlpack({kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION});

    auto new_double_array = double_array.from_dlpack(double_dl_tensor);
    // create an array of a different type from the source array
    auto new_int_array = double_array.from_dlpack(int_dl_tensor);

    auto device = DLDevice{kDLCPU, 0};
    auto version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};

    CHECK(
        new_double_array.as_dlpack_array<double>(device, nullptr, version)
        == double_array.as_dlpack_array<double>(device, nullptr, version)
    );

    CHECK(
        new_int_array.as_dlpack_array<int16_t>(device, nullptr, version)
        == int_array.as_dlpack_array<int16_t>(device, nullptr, version)
    );
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

TEST_CASE("SimpleDataArray - dtype()") {
    SECTION("double") {
        auto data = SimpleDataArray<double>({2, 3});
        auto dt = data.dtype();
        CHECK(dt.code == kDLFloat);
        CHECK(dt.bits == 64);
        CHECK(dt.lanes == 1);
    }

    SECTION("float") {
        auto data = SimpleDataArray<float>({2, 3});
        auto dt = data.dtype();
        CHECK(dt.code == kDLFloat);
        CHECK(dt.bits == 32);
        CHECK(dt.lanes == 1);
    }

    SECTION("int32") {
        auto data = SimpleDataArray<int32_t>({2, 3});
        auto dt = data.dtype();
        CHECK(dt.code == kDLInt);
        CHECK(dt.bits == 32);
        CHECK(dt.lanes == 1);
    }

    SECTION("via mts_array_t callback") {
        auto data = std::make_unique<SimpleDataArray<double>>(SimpleDataArray<double>({2, 3}));
        auto array = DataArrayBase::to_mts_array(std::move(data));

        CHECK(array.dtype().code == kDLFloat);
        CHECK(array.dtype().bits == 64);
        CHECK(array.dtype().lanes == 1);
    }
}

TEST_CASE("EmptyDataArray - dtype()") {
    SECTION("default (double)") {
        auto data = EmptyDataArray<double>({2, 3});
        auto dt = data.dtype();
        CHECK(dt.code == kDLFloat);
        CHECK(dt.bits == 64);
        CHECK(dt.lanes == 1);
    }

    SECTION("int32") {
        auto data = EmptyDataArray<int32_t>({2, 3});
        auto dt = data.dtype();
        CHECK(dt.code == kDLInt);
        CHECK(dt.bits == 32);
        CHECK(dt.lanes == 1);
    }

    SECTION("bool") {
        auto data = EmptyDataArray<bool>({2, 3});
        auto dt = data.dtype();
        CHECK(dt.code == kDLBool);
        CHECK(dt.bits == 8);
        CHECK(dt.lanes == 1);
    }

    SECTION("via mts_array_t callback") {
        auto data = std::make_unique<EmptyDataArray<int32_t>>(std::vector<uintptr_t>{2, 3});
        auto array = DataArrayBase::to_mts_array(std::move(data));

        CHECK(array.dtype().code == kDLInt);
        CHECK(array.dtype().bits == 32);
        CHECK(array.dtype().lanes == 1);
    }
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


TEST_CASE("default_create_array with bool dtype") {
    auto dtype = metatensor::details::dtype_of<bool>();
    CHECK(dtype.code == kDLBool);
    CHECK(dtype.bits == 8);
    CHECK(dtype.lanes == 1);

    std::vector<uintptr_t> shape = {2, 3};

    mts_array_t raw_array = {};
    auto status = metatensor::details::default_create_array(
        shape.data(), shape.size(), dtype, &raw_array
    );
    CHECK(status == MTS_SUCCESS);

    auto array = MtsArray(raw_array);

    CHECK(array.shape() == shape);

    auto dt = array.dtype();
    CHECK(dt.code == kDLBool);
    CHECK(dt.bits == 8);
    CHECK(dt.lanes == 1);

    // The newly created array should be zero-initialised
    auto version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto dlpack = DLPackArray<bool>(array.as_dlpack({kDLCPU, 0}, nullptr, version));

    CHECK(dlpack.shape() == std::vector<size_t>{2, 3});
    for (size_t i = 0; i < 6; i++) {
        CHECK(dlpack.data()[i] == false);
    }
}


TEST_CASE("SimpleDataArray<bool>") {
    SECTION("from_dlpack() with bool") {
        auto raw_data = std::make_shared<std::vector<uint8_t>>(
            std::vector<uint8_t>{1, 0, 0, 1, 0, 1}
        );
        auto* dl_managed = make_bool_dlpack(raw_data, {2, 3});

        // Use a SimpleDataArray<double> as the "source" array (its from_dlpack
        // is the static dispatcher that handles all dtypes)
        auto source = std::make_unique<SimpleDataArray<double>>(SimpleDataArray<double>({1}));
        auto source_array = DataArrayBase::to_mts_array(std::move(source));

        auto new_array = source_array.from_dlpack(dl_managed);

        auto* base = static_cast<DataArrayBase*>(new_array.as_mts_array_t().ptr);
        REQUIRE(dynamic_cast<SimpleDataArray<bool>*>(base) != nullptr);

        CHECK(new_array.shape() == std::vector<uintptr_t>{2, 3});

        auto dt = new_array.dtype();
        CHECK(dt.code == kDLBool);
        CHECK(dt.bits == 8);
        CHECK(dt.lanes == 1);

        // Verify the data round-trips through DLPack
        auto version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        auto dlpack = DLPackArray<bool>(new_array.as_dlpack({kDLCPU, 0}, nullptr, version));

        CHECK(dlpack.shape() == std::vector<size_t>{2, 3});
        CHECK(dlpack.data()[0] == true);
        CHECK(dlpack.data()[1] == false);
        CHECK(dlpack.data()[2] == false);
        CHECK(dlpack.data()[3] == true);
        CHECK(dlpack.data()[4] == false);
        CHECK(dlpack.data()[5] == true);
    }

    SECTION("round-trip through DLPack") {
        // Start from a bool DLPack tensor, import via from_dlpack, export back
        // via as_dlpack, and verify dtype + values are preserved.
        auto raw_data = std::make_shared<std::vector<uint8_t>>(
            std::vector<uint8_t>{0, 1, 1, 0, 1, 0, 0, 1}
        );
        auto* dl_managed = make_bool_dlpack(raw_data, {2, 4});

        auto source = std::make_unique<SimpleDataArray<double>>(SimpleDataArray<double>({1}));
        auto source_array = DataArrayBase::to_mts_array(std::move(source));

        auto bool_array = source_array.from_dlpack(dl_managed);

        // Export back to DLPack
        auto version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        auto* re_exported = bool_array.as_dlpack({kDLCPU, 0}, nullptr, version);

        CHECK(re_exported->dl_tensor.dtype.code == kDLBool);
        CHECK(re_exported->dl_tensor.dtype.bits == 8);
        CHECK(re_exported->dl_tensor.dtype.lanes == 1);

        // Verify data is preserved
        const auto* ptr = static_cast<const uint8_t*>(re_exported->dl_tensor.data);
        CHECK(ptr[0] == 0);
        CHECK(ptr[1] == 1);
        CHECK(ptr[2] == 1);
        CHECK(ptr[3] == 0);
        CHECK(ptr[4] == 1);
        CHECK(ptr[5] == 0);
        CHECK(ptr[6] == 0);
        CHECK(ptr[7] == 1);

        re_exported->deleter(re_exported);
    }
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
