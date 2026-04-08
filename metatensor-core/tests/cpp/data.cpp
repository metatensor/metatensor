#include <cstring>
#include <memory>

#include <catch.hpp>

#include <metatensor.hpp>
using namespace metatensor;

TEST_CASE("Data Array") {
    auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3, 4}));
    auto array = DataArrayBase::to_mts_array_t(std::move(data));

    SECTION("origin") {
        mts_data_origin_t origin = 0;
        auto status = array.origin(array.ptr, &origin);
        CHECK(status == MTS_SUCCESS);

        char buffer[64] = {0};
        status = mts_get_data_origin(origin, buffer, 64);
        CHECK(status == MTS_SUCCESS);
        CHECK(std::string(buffer) == "metatensor::SimpleDataArray");
    }

    SECTION("shape") {
        const uintptr_t* shape = nullptr;
        uintptr_t shape_count = 0;
        auto status = array.shape(array.ptr, &shape, &shape_count);
        CHECK(status == MTS_SUCCESS);

        CHECK(shape_count == 3);
        CHECK(shape[0] == 2);
        CHECK(shape[1] == 3);
        CHECK(shape[2] == 4);

        uintptr_t new_shape[] = {1, 2, 3, 4};
        shape_count = 4;
        status = array.reshape(array.ptr, new_shape, shape_count);
        CHECK(status == MTS_SUCCESS);

        status = array.shape(array.ptr, &shape, &shape_count);
        CHECK(status == MTS_SUCCESS);

        CHECK(shape_count == 4);
        CHECK(shape[0] == 1);
        CHECK(shape[1] == 2);
        CHECK(shape[2] == 3);
        CHECK(shape[3] == 4);

        status = array.swap_axes(array.ptr, 1, 2);
        CHECK(status == MTS_SUCCESS);

        status = array.shape(array.ptr, &shape, &shape_count);
        CHECK(status == MTS_SUCCESS);

        CHECK(shape_count == 4);
        CHECK(shape[0] == 1);
        CHECK(shape[1] == 3);
        CHECK(shape[2] == 2);
        CHECK(shape[3] == 4);
    }

    SECTION("new arrays") {
        mts_array_t new_array;
        std::memset(&new_array, 0, sizeof(new_array));
        auto status = array.copy(array.ptr, &new_array);
        CHECK(status == MTS_SUCCESS);


        const uintptr_t* shape = nullptr;
        uintptr_t shape_count = 0;
        status = new_array.shape(new_array.ptr, &shape, &shape_count);
        CHECK(status == MTS_SUCCESS);

        CHECK(shape_count == 3);
        CHECK(shape[0] == 2);
        CHECK(shape[1] == 3);
        CHECK(shape[2] == 4);
        new_array.destroy(new_array.ptr);

        uintptr_t new_shape[] = {1, 2, 3, 4};
        shape_count = 4;
        auto fill_value = DataArrayBase::to_mts_array_t(
            std::make_unique<SimpleDataArray<double>>(std::vector<uintptr_t>{}, 0.0)
        );
        status = array.create(array.ptr, new_shape, shape_count, fill_value, &new_array);
        CHECK(status == MTS_SUCCESS);

        status = new_array.shape(new_array.ptr, &shape, &shape_count);
        CHECK(status == MTS_SUCCESS);

        CHECK(shape_count == 4);
        CHECK(shape[0] == 1);
        CHECK(shape[1] == 2);
        CHECK(shape[2] == 3);
        CHECK(shape[3] == 4);
        new_array.destroy(new_array.ptr);
    }

    array.destroy(array.ptr);
}

TEST_CASE("SimpleDataArray<double> - as_dlpack()") {
    auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3, 4}));
    auto array = DataArrayBase::to_mts_array_t(std::move(data));

    // mutate via view
    {
        auto view = static_cast<SimpleDataArray<double>*>(array.ptr)->view();
        view(1, 1, 0) = 1.2345;
    }

    // as_dlpack -> check dtype and data
    {
        auto *s = static_cast<SimpleDataArray<double>*>(array.ptr);
        DLDevice cpu_device = {kDLCPU, 0};
        DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        DLManagedTensorVersioned* managed = s->as_dlpack(cpu_device, nullptr, ver);
        REQUIRE(managed != nullptr);

        const DLTensor& t = managed->dl_tensor;
        CHECK(t.device.device_type == kDLCPU);
        CHECK(t.device.device_id == 0);
        CHECK(t.ndim == 3);
        REQUIRE(t.shape != nullptr);
        REQUIRE(t.strides != nullptr);
        CHECK(t.shape[0] == 2);
        CHECK(t.shape[1] == 3);
        CHECK(t.shape[2] == 4);
        CHECK(t.strides[0] == 12);
        CHECK(t.strides[1] == 4);
        CHECK(t.strides[2] == 1);

        CHECK(t.dtype.code == kDLFloat);
        CHECK(t.dtype.bits == 64);
        CHECK(t.dtype.lanes == 1);

        REQUIRE(t.data != nullptr);
        double* pdata = static_cast<double*>(t.data);
        CHECK(pdata[16] == Approx(1.2345));

        // free managed tensor using its deleter (should not crash)
        managed->deleter(managed);
    }

    array.destroy(array.ptr);
}

TEST_CASE("SimpleDataArray<float> - as_dlpack()") {
    // Create float-backed array via C++ class, expose through mts_array_t
    auto data = std::unique_ptr<SimpleDataArray<float>>(new SimpleDataArray<float>({2, 3, 4}));
    auto array = DataArrayBase::to_mts_array_t(std::move(data));

    // as_dlpack should provide a valid float-typed DLPack view
    {
        auto *s = static_cast<SimpleDataArray<float>*>(array.ptr);
        auto view = s->view();
        view(1, 1, 0) = static_cast<float>(3.1415);

        DLDevice cpu_device = {kDLCPU, 0};
        DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        DLManagedTensorVersioned* managed = s->as_dlpack(cpu_device, nullptr, ver);
        REQUIRE(managed != nullptr);

        const DLTensor& t = managed->dl_tensor;
        CHECK(t.dtype.code == kDLFloat);
        CHECK(t.dtype.bits == 32);
        CHECK(t.dtype.lanes == 1);

        REQUIRE(t.data != nullptr);
        auto* pdata = static_cast<float*>(t.data);
        CHECK(pdata[16] == static_cast<float>(3.1415));

        managed->deleter(managed);
    }

    array.destroy(array.ptr);
}

TEST_CASE("SimpleDataArray<int32_t> - as_dlpack()") {
    auto data = std::unique_ptr<SimpleDataArray<int32_t>>(new SimpleDataArray<int32_t>({2, 3, 4}));
    auto array = DataArrayBase::to_mts_array_t(std::move(data));

    // DLPack view should be int32
    {
        auto *s = static_cast<SimpleDataArray<int32_t>*>(array.ptr);
        auto view = s->view();
        view(1, 1, 0) = 42;

        DLDevice cpu_device = {kDLCPU, 0};
        DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        DLManagedTensorVersioned* managed = s->as_dlpack(cpu_device, nullptr, ver);
        REQUIRE(managed != nullptr);

        const DLTensor& t = managed->dl_tensor;
        CHECK(t.dtype.code == kDLInt);
        CHECK(t.dtype.bits == 32);
        CHECK(t.dtype.lanes == 1);

        REQUIRE(t.data != nullptr);
        auto* pdata = static_cast<int32_t*>(t.data);
        CHECK(pdata[16] == 42);

        managed->deleter(managed);
    }

    array.destroy(array.ptr);
}

TEST_CASE("DLPackArray<T> - construction and access") {
    // Create a SimpleDataArray<double>, get a DLPack tensor, wrap in DLPackArray
    auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3}));
    auto view = data->view();
    view(0, 0) = 1.0;
    view(0, 1) = 2.0;
    view(0, 2) = 3.0;
    view(1, 0) = 4.0;
    view(1, 1) = 5.0;
    view(1, 2) = 6.0;

    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    DLManagedTensorVersioned* managed = data->as_dlpack(cpu_device, nullptr, ver);
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
    auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3}));

    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    DLManagedTensorVersioned* managed = data->as_dlpack(cpu_device, nullptr, ver);
    REQUIRE(managed != nullptr);

    // double tensor should NOT match int32_t
    CHECK_THROWS_WITH(
        DLPackArray<int32_t>(managed),
        Catch::Matchers::Contains("dtype mismatch")
    );

    // After dtype mismatch, the managed tensor should have been cleaned up.
    // Get a new one for float mismatch test
    managed = data->as_dlpack(cpu_device, nullptr, ver);
    REQUIRE(managed != nullptr);

    CHECK_THROWS_WITH(
        DLPackArray<float>(managed),
        Catch::Matchers::Contains("dtype mismatch")
    );
}

TEST_CASE("DLPackArray<T> - with int32 data") {
    auto data = std::unique_ptr<SimpleDataArray<int32_t>>(new SimpleDataArray<int32_t>({2, 2}));
    auto view = data->view();
    view(0, 0) = 10;
    view(0, 1) = 20;
    view(1, 0) = 30;
    view(1, 1) = 40;

    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    DLManagedTensorVersioned* managed = data->as_dlpack(cpu_device, nullptr, ver);
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

    SECTION("direct call") {
        auto dev = data->device();
        CHECK(dev.device_type == kDLCPU);
        CHECK(dev.device_id == 0);
    }

    SECTION("via mts_array_t callback") {
        auto array = DataArrayBase::to_mts_array_t(std::move(data));
        REQUIRE(array.device != nullptr);

        DLDevice dev;
        std::memset(&dev, 0xFF, sizeof(dev));
        auto status = array.device(array.ptr, &dev);
        CHECK(status == MTS_SUCCESS);
        CHECK(dev.device_type == kDLCPU);
        CHECK(dev.device_id == 0);

        array.destroy(array.ptr);
    }
}

TEST_CASE("EmptyDataArray - device()") {
    auto data = std::unique_ptr<EmptyDataArray>(new EmptyDataArray({2, 3}));
    auto dev = data->device();
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
        auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3}));
        DLDevice cpu_device = {kDLCPU, 0};
        DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        DLManagedTensorVersioned* managed = data->as_dlpack(cpu_device, nullptr, ver);
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
    auto array = DataArrayBase::to_mts_array_t(std::move(data));
    auto *s = static_cast<SimpleDataArray<double>*>(array.ptr);

    DLDevice cpu_device = {kDLCPU, 0};

    // Case 1: Request an older, incompatible version (0.1)
    // The implementation supports 1.1, so this must fail.
    DLPackVersion old_version = {0, 1};
    CHECK_THROWS_WITH(
        s->as_dlpack(cpu_device, nullptr, old_version),
        Catch::Matchers::Contains("SimpleDataArray supports DLPack version")
    );

    // Case 2: Request a newer version (1.5)
    // Succeed because 1.5 is compatible with 1.3
    DLPackVersion new_version = {1, 5};
    DLManagedTensorVersioned* managed = nullptr;
    CHECK_NOTHROW(managed = s->as_dlpack(cpu_device, nullptr, new_version));

    // Case 3: Request an ABI breaking version (2.0)
    // Succeed because 1.5 is compatible with 1.0
    DLPackVersion too_new_version = {2, 0};
    CHECK_THROWS_WITH(
        s->as_dlpack(cpu_device, nullptr, too_new_version),
        Catch::Matchers::Contains("Caller requested incompatible version")
    );

    // Verify we got back the implementation version, not the requested one
    REQUIRE(managed != nullptr);
    CHECK(managed->version.major == DLPACK_MAJOR_VERSION);
    CHECK(managed->version.minor <= 3);

    managed->deleter(managed);
    array.destroy(array.ptr);
}

TEST_CASE("SimpleDataArray - dtype()") {
    SECTION("double") {
        auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3}));
        auto dt = data->dtype();
        CHECK(dt.code == kDLFloat);
        CHECK(dt.bits == 64);
        CHECK(dt.lanes == 1);
    }

    SECTION("float") {
        auto data = std::unique_ptr<SimpleDataArray<float>>(new SimpleDataArray<float>({2, 3}));
        auto dt = data->dtype();
        CHECK(dt.code == kDLFloat);
        CHECK(dt.bits == 32);
        CHECK(dt.lanes == 1);
    }

    SECTION("int32") {
        auto data = std::unique_ptr<SimpleDataArray<int32_t>>(new SimpleDataArray<int32_t>({2, 3}));
        auto dt = data->dtype();
        CHECK(dt.code == kDLInt);
        CHECK(dt.bits == 32);
        CHECK(dt.lanes == 1);
    }

    SECTION("via mts_array_t callback") {
        auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3}));
        auto array = DataArrayBase::to_mts_array_t(std::move(data));
        REQUIRE(array.dtype != nullptr);

        DLDataType dt;
        std::memset(&dt, 0xFF, sizeof(dt));
        auto status = array.dtype(array.ptr, &dt);
        CHECK(status == MTS_SUCCESS);
        CHECK(dt.code == kDLFloat);
        CHECK(dt.bits == 64);
        CHECK(dt.lanes == 1);

        array.destroy(array.ptr);
    }
}

TEST_CASE("EmptyDataArray - dtype()") {
    auto data = std::unique_ptr<EmptyDataArray>(new EmptyDataArray({2, 3}));
    auto dt = data->dtype();
    CHECK(dt.code == kDLFloat);
    CHECK(dt.bits == 64);
    CHECK(dt.lanes == 1);
}
