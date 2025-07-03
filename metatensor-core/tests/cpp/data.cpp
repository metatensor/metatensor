#include <vector>

#include <catch.hpp>

// This must be included for the DLManagedTensor definition
#include <dlpack/dlpack.h>
#include <metatensor.hpp>

using namespace metatensor;

// A mock DataArray that can produce a DLPack tensor. This is used to test
// the C-API layer directly, bypassing the C++ DataArrayBase wrapper.
// xref: gh-934
class DlpackDataArray {
      public:
        // Helper struct to manage the lifetime of all data associated with the
        // DLManagedTensor. The `manager_ctx` will hold a pointer to an instance
        // of this struct.
        struct DlpackData {
                // The actual tensor data
                std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
                // The shape of the tensor, stored as int64_t for DLPack
                std::vector<int64_t> shape_dlpack = {2, 3};
                // The DLManagedTensor struct itself. This will be returned to
                // the consumer, but its lifetime is tied to this DlpackData
                // struct.
                DLManagedTensor tensor;
        };
        // The shape of the tensor, stored as uintptr_t for metatensor's shape
        // function
        std::vector<uintptr_t> shape_uintptr = {2, 3};

        DlpackDataArray() = default;
};

// Deleter function for the DLManagedTensor, as required by the DLPack standard
static void dlpack_deleter(DLManagedTensor *self) {
        if (self) {
                // The manager_ctx holds the pointer to the DlpackData struct
                // that owns all the memory. Deleting it cleans up everything.
                delete static_cast<DlpackDataArray::DlpackData *>(
                    self->manager_ctx);
        }
}

// C-style callback function that implements to_dlpack for DlpackDataArray
static mts_status_t dlpack_data_array_to_dlpack(const void *array,
                                                DLManagedTensor **dl_tensor) {
        try {
                // Create the data holder on the heap. Its lifetime will be
                // managed by the DLManagedTensor through the deleter.
                auto dl_data = new DlpackDataArray::DlpackData();

                // Fill the DLTensor fields
                dl_data->tensor.dl_tensor.data = dl_data->data.data();
                dl_data->tensor.dl_tensor.device = {kDLCPU, 0};
                dl_data->tensor.dl_tensor.ndim = 2;
                dl_data->tensor.dl_tensor.dtype = {kDLFloat, 32, 1};
                dl_data->tensor.dl_tensor.shape = dl_data->shape_dlpack.data();
                dl_data->tensor.dl_tensor.strides =
                    nullptr; // compact row-major
                dl_data->tensor.dl_tensor.byte_offset = 0;

                // Fill the DLManagedTensor fields
                dl_data->tensor.manager_ctx = dl_data;
                dl_data->tensor.deleter = &dlpack_deleter;

                *dl_tensor = &dl_data->tensor;
        } catch (...) {
                return MTS_INTERNAL_ERROR;
        }
        return MTS_SUCCESS;
}

// C-style callback for origin
static mts_status_t dlpack_data_array_origin(const void *array,
                                             mts_data_origin_t *origin) {
        return mts_register_data_origin("metatensor::DlpackDataArray", origin);
}

// C-style callback for shape
static mts_status_t dlpack_data_array_shape(const void *array,
                                            const uintptr_t **shape,
                                            uintptr_t *shape_count) {
        auto self = static_cast<const DlpackDataArray *>(array);
        *shape = self->shape_uintptr.data();
        *shape_count = self->shape_uintptr.size();
        return MTS_SUCCESS;
}

// C-style callback for destroy
static void dlpack_data_array_destroy(void *array) {
        delete static_cast<DlpackDataArray *>(array);
}

TEST_CASE("Data Array") {
        auto data =
            std::unique_ptr<SimpleDataArray>(new SimpleDataArray({2, 3, 4}));
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
                auto view = static_cast<SimpleDataArray *>(array.ptr)->view();
                view(1, 1, 0) = 3;

                double *data_ptr = nullptr;
                auto status = array.data(array.ptr, &data_ptr);
                CHECK(status == MTS_SUCCESS);
                CHECK(data_ptr[0] == 0);
                CHECK(data_ptr[16] == 3);
        }

        SECTION("shape") {
                const uintptr_t *shape = nullptr;
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

                const uintptr_t *shape = nullptr;
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
                status =
                    array.create(array.ptr, new_shape, shape_count, &new_array);
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

TEST_CASE("DLPack Array") {
        auto dlpack_data = new DlpackDataArray();
        mts_array_t dlpack_array = {
            .ptr = static_cast<void *>(dlpack_data),
            .origin = &dlpack_data_array_origin,
            .data = nullptr,
            .to_dlpack = &dlpack_data_array_to_dlpack,
            .shape = &dlpack_data_array_shape,
            .reshape = nullptr,
            .swap_axes = nullptr,
            .create = nullptr,
            .copy = nullptr,
            .destroy = &dlpack_data_array_destroy,
            .move_samples_from = nullptr,
        };
        DLManagedTensor *dl_tensor = nullptr;

        // Existence check
        auto status = dlpack_array.to_dlpack(dlpack_array.ptr, &dl_tensor);
        CHECK(status == MTS_SUCCESS);
        REQUIRE(dl_tensor != nullptr);

        SECTION("DLPack metadata") {
                // Check the tensor properties
                CHECK(dl_tensor->dl_tensor.device.device_type == kDLCPU);
                CHECK(dl_tensor->dl_tensor.ndim == 2);
                CHECK(dl_tensor->dl_tensor.dtype.code == kDLFloat);
                CHECK(dl_tensor->dl_tensor.dtype.bits == 32);
                CHECK(dl_tensor->dl_tensor.dtype.lanes == 1);
                CHECK(dl_tensor->dl_tensor.shape[0] == 2);
                CHECK(dl_tensor->dl_tensor.shape[1] == 3);
                CHECK(dl_tensor->dl_tensor.strides == nullptr);
                CHECK(dl_tensor->dl_tensor.byte_offset == 0);
        }

        SECTION("Data validity (float)") {
                auto *data_ptr =
                    static_cast<float *>(dl_tensor->dl_tensor.data);
                CHECK(data_ptr[0] == 1.0f);
                CHECK(data_ptr[5] == 6.0f);
        }
        // The consumer is responsible for calling the deleter to free
        // the memory
        dl_tensor->deleter(dl_tensor);

        // The original array can now be destroyed
        dlpack_array.destroy(dlpack_array.ptr);
}
