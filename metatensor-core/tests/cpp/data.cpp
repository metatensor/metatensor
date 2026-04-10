#include <cstring>
#include <memory>

#include <catch.hpp>

#include <metatensor.hpp>
using namespace metatensor;

TEST_CASE("Data Array") {
    auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3, 4}));
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
    auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3, 4}));
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
    auto data = std::unique_ptr<SimpleDataArray<float>>(new SimpleDataArray<float>({2, 3, 4}));
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
    auto data = std::unique_ptr<SimpleDataArray<int32_t>>(new SimpleDataArray<int32_t>({2, 3, 4}));
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
        CHECK(arr.is_empty() == false);

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

    SECTION("move construction") {
        auto arr1 = DLPackArray<double>(managed);
        auto arr2 = DLPackArray<double>(std::move(arr1));

        CHECK(arr2.shape().size() == 2);
        CHECK(arr2(1, 2) == 6.0);

        // arr1 should be empty after move
        CHECK(arr1.shape() == std::vector<size_t>{0, 0});
        CHECK(arr1.data() == nullptr);
        CHECK(arr1.is_empty() == true);
    }

    SECTION("move assignment") {
        auto arr1 = DLPackArray<double>(managed);
        auto arr2 = DLPackArray<double>();

        arr2 = std::move(arr1);
        CHECK(arr2(0, 0) == 1.0);
        CHECK(arr1.is_empty() == true);
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
    auto arr = DLPackArray<double>();
    CHECK(arr.is_empty() == true);
    CHECK(arr.data() == nullptr);
    CHECK(arr.shape() == std::vector<size_t>{0, 0});
}

TEST_CASE("DLPackArray<T> - nullptr construction") {
    auto arr = DLPackArray<double>(nullptr);
    CHECK(arr.is_empty() == true);
    CHECK(arr.data() == nullptr);
}

TEST_CASE("SimpleDataArray - device()") {
    auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3}));

    // direct call to device()
    CHECK(data->device().device_type == kDLCPU);
    CHECK(data->device().device_id == 0);

    // through mts_array_t callback
    auto array = DataArrayBase::to_mts_array(std::move(data));
    CHECK(array.device().device_type == kDLCPU);
    CHECK(array.device().device_id == 0);
}

TEST_CASE("EmptyDataArray - device()") {
    auto data = EmptyDataArray({2, 3});
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
    auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3}, 1.0));
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
    auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 2}));

    DLDevice cpu_device = {kDLCPU, 0};

    // Case 1: Request an older, incompatible version (0.1)
    // The implementation supports 1.1, so this must fail.
    DLPackVersion old_version = {0, 1};
    CHECK_THROWS_WITH(
        data->as_dlpack(cpu_device, nullptr, old_version),
        Catch::Matchers::Contains("SimpleDataArray supports DLPack version")
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
        Catch::Matchers::Contains("Caller requested incompatible version")
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
        auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3}));
        auto array = DataArrayBase::to_mts_array(std::move(data));

        CHECK(array.dtype().code == kDLFloat);
        CHECK(array.dtype().bits == 64);
        CHECK(array.dtype().lanes == 1);
    }
}

TEST_CASE("EmptyDataArray - dtype()") {
    auto data = EmptyDataArray({2, 3});
    auto dt = data.dtype();
    CHECK(dt.code == kDLFloat);
    CHECK(dt.bits == 64);
    CHECK(dt.lanes == 1);
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
