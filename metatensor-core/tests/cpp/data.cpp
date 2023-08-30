#include <memory>

#include <catch.hpp>

#include <metatensor.hpp>
using namespace metatensor;

TEST_CASE("Data Array") {
    auto data = std::unique_ptr<SimpleDataArray>(new SimpleDataArray({2, 3, 4}));
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
        auto view = static_cast<SimpleDataArray*>(array.ptr)->view();
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
