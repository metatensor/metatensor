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

    SECTION("data") {
        auto view = static_cast<SimpleDataArray<double>*>(array.ptr)->view();
        view(1, 1, 0) = 3;

        double* data_ptr = nullptr;
        auto status = array.data(array.ptr, &data_ptr);
        CHECK(status == MTS_SUCCESS);
        CHECK(data_ptr[0] == 0);
        CHECK(data_ptr[16] == 3);
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
        status = array.create(array.ptr, new_shape, shape_count, &new_array);
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

TEST_CASE("SimpleDataArray<double) - data() and as_dlpack()") {
    auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 3, 4}));
    auto array = DataArrayBase::to_mts_array_t(std::move(data));

    // mutate via view and check C API data() works for double
    {
        auto view = static_cast<SimpleDataArray<double>*>(array.ptr)->view();
        view(1, 1, 0) = 1.2345;

        double* data_ptr = nullptr;
        auto status = array.data(array.ptr, &data_ptr);
        CHECK(status == MTS_SUCCESS);
        REQUIRE(data_ptr != nullptr);
        // flattened index (1,1,0) = 1*(3*4) + 1*4 + 0 = 16
        CHECK(data_ptr[16] == Approx(1.2345));
    }

    // as_dlpack -> check dtype and data
    {
        auto s = static_cast<SimpleDataArray<double>*>(array.ptr);
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
        CHECK(t.dtype.bits == 64u);
        CHECK(t.dtype.lanes == 1u);

        REQUIRE(t.data != nullptr);
        double* pdata = static_cast<double*>(t.data);
        CHECK(pdata[16] == Approx(1.2345));

        // free managed tensor using its deleter (should not crash)
        managed->deleter(managed);
    }

    array.destroy(array.ptr);
}

TEST_CASE("SimpleDataArray<float) - as_dlpack() and C API data() refusal") {
    // Create float-backed array via C++ class, expose through mts_array_t
    auto data = std::unique_ptr<SimpleDataArray<float>>(new SimpleDataArray<float>({2, 3, 4}));
    auto array = DataArrayBase::to_mts_array_t(std::move(data));

    // C API data() expects double* and should fail for float-backed arrays
    {
        double* data_ptr = nullptr;
        auto status = array.data(array.ptr, &data_ptr);
        CHECK(status != MTS_SUCCESS); // should not succeed
    }

    // But as_dlpack should provide a valid float-typed DLPack view
    {
        auto s = static_cast<SimpleDataArray<float>*>(array.ptr);
        auto view = s->view();
        view(1, 1, 0) = 3.1415f;

        DLDevice cpu_device = {kDLCPU, 0};
        DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        DLManagedTensorVersioned* managed = s->as_dlpack(cpu_device, nullptr, ver);
        REQUIRE(managed != nullptr);

        const DLTensor& t = managed->dl_tensor;
        CHECK(t.dtype.code == kDLFloat);
        CHECK(t.dtype.bits == 32u);
        CHECK(t.dtype.lanes == 1u);

        REQUIRE(t.data != nullptr);
        float* pdata = static_cast<float*>(t.data);
        CHECK(pdata[16] == Approx(3.1415f));

        managed->deleter(managed);
    }

    array.destroy(array.ptr);
}

TEST_CASE("SimpleDataArray<int32_t) - as_dlpack() and C API data() refusal") {
    auto data = std::unique_ptr<SimpleDataArray<int32_t>>(new SimpleDataArray<int32_t>({2, 3, 4}));
    auto array = DataArrayBase::to_mts_array_t(std::move(data));

    // C API should refuse to return double* for int32 array
    {
        double* data_ptr = nullptr;
        auto status = array.data(array.ptr, &data_ptr);
        CHECK(status != MTS_SUCCESS);
    }

    // DLPack view should be int32
    {
        auto s = static_cast<SimpleDataArray<int32_t>*>(array.ptr);
        auto view = s->view();
        view(1, 1, 0) = 42;

        DLDevice cpu_device = {kDLCPU, 0};
        DLPackVersion ver = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        DLManagedTensorVersioned* managed = s->as_dlpack(cpu_device, nullptr, ver);
        REQUIRE(managed != nullptr);

        const DLTensor& t = managed->dl_tensor;
        CHECK(t.dtype.code == kDLInt);
        CHECK(t.dtype.bits == 32u);
        CHECK(t.dtype.lanes == 1u);

        REQUIRE(t.data != nullptr);
        int32_t* pdata = static_cast<int32_t*>(t.data);
        CHECK(pdata[16] == 42);

        managed->deleter(managed);
    }

    array.destroy(array.ptr);
}

TEST_CASE("SimpleDataArray - DLPack version mismatch") {
    auto data = std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 2}));
    auto array = DataArrayBase::to_mts_array_t(std::move(data));
    auto s = static_cast<SimpleDataArray<double>*>(array.ptr);

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
