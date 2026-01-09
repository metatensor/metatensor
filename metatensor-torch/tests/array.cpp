#include <torch/torch.h>

#include <metatensor/torch.hpp>
using namespace metatensor_torch;

#include <catch.hpp>

TEST_CASE("Arrays") {
    auto tensor = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kF64));
    auto array = TorchDataArray(tensor);

    SECTION("origin") {
        auto origin = array.origin();

        std::array<char, 64> buffer = {0};
        auto status = mts_get_data_origin(origin, buffer.data(), buffer.size());
        CHECK(status == MTS_SUCCESS);
        CHECK(std::string(buffer.data()) == "metatensor_torch::TorchDataArray");
    }

    SECTION("shape") {
        auto shape = array.shape();
        CHECK(shape.size() == 3);
        CHECK(shape[0] == 2);
        CHECK(shape[1] == 3);
        CHECK(shape[2] == 4);

        CHECK((array.tensor().sizes() == std::vector<int64_t>{2, 3, 4}));

        array.reshape({1, 2, 3, 4});
        shape = array.shape();
        CHECK(shape.size() == 4);
        CHECK(shape[0] == 1);
        CHECK(shape[1] == 2);
        CHECK(shape[2] == 3);
        CHECK(shape[3] == 4);

        CHECK((array.tensor().sizes() == std::vector<int64_t>{1, 2, 3, 4}));

        array.swap_axes(1, 2);
        shape = array.shape();
        CHECK(shape.size() == 4);
        CHECK(shape[0] == 1);
        CHECK(shape[1] == 3);
        CHECK(shape[2] == 2);
        CHECK(shape[3] == 4);

        CHECK((array.tensor().sizes() == std::vector<int64_t>{1, 3, 2, 4}));
    }

    SECTION("new arrays") {
        auto copy = array.copy();
        auto* copy_ptr = dynamic_cast<TorchDataArray*>(copy.get());

        CHECK(copy_ptr->tensor().data_ptr() != array.tensor().data_ptr());
        CHECK((copy_ptr->tensor().sizes() == std::vector<int64_t>{2, 3, 4}));
        CHECK(copy_ptr->tensor().dtype() == torch::kF64);

        auto created = array.create({5, 6});
        auto* created_ptr = dynamic_cast<TorchDataArray*>(created.get());

        CHECK((created_ptr->tensor().sizes() == std::vector<int64_t>{5, 6}));
        CHECK(created_ptr->tensor().dtype() == torch::kF64);
    }

    SECTION("DLPack conversion") {
        // Common
        DLDevice cpu_device = {kDLCPU, 0};
        DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};

        // Basic Conversion & Metadata Check
        {
            auto dl_managed = array.as_dlpack(cpu_device, nullptr, version);

            // Check version
            CHECK(dl_managed->version.major == DLPACK_MAJOR_VERSION);
            CHECK(dl_managed->version.minor == DLPACK_MINOR_VERSION);

            // Check DLTensor fields
            auto &dl_tensor = dl_managed->dl_tensor;
            CHECK(dl_tensor.ndim == 3);
            CHECK(dl_tensor.shape[0] == 2);
            CHECK(dl_tensor.shape[1] == 3);
            CHECK(dl_tensor.shape[2] == 4);

            // Check Data Type (kF64 -> kDLFloat, 64 bits)
            CHECK(dl_tensor.dtype.code == kDLFloat);
            CHECK(dl_tensor.dtype.bits == 64);
            CHECK(dl_tensor.dtype.lanes == 1);

            // Check Data Pointer Equality
            // The DLPack pointer should point to the same memory as the Torch
            // tensor
            CHECK(dl_tensor.data == array.tensor().data_ptr());

            dl_managed->deleter(dl_managed);
        }

        // Different data types (e.g., Int32)
        {
            auto int_tensor = torch::tensor(
                {1, 2, 3}, torch::TensorOptions().dtype(torch::kInt32));
            auto int_array = TorchDataArray(int_tensor);
            auto dl_managed = int_array.as_dlpack(cpu_device, nullptr, version);

            CHECK(dl_managed->dl_tensor.dtype.code == kDLInt);
            CHECK(dl_managed->dl_tensor.dtype.bits == 32);

            dl_managed->deleter(dl_managed);
        }

        // Stride verification (Non-contiguous memory)
        {
            // Create a non-contiguous tensor by slicing
            auto base_tensor =
                torch::zeros({10, 10}, torch::TensorOptions().dtype(torch::kF64));
            // Take every 2nd element along both dimensions
            auto sliced_tensor = base_tensor.slice(0, 0, 10, 2).slice(1, 0, 10, 2);
            auto sliced_array = TorchDataArray(sliced_tensor);

            auto dl_managed = sliced_array.as_dlpack(cpu_device, nullptr, version);
            auto &dl_tensor = dl_managed->dl_tensor;

            CHECK(dl_tensor.ndim == 2);
            CHECK(dl_tensor.shape[0] == 5);
            CHECK(dl_tensor.shape[1] == 5);

            // Torch strides are usually (stride_0, stride_1).
            // For a 10x10 tensor, row stride is 10, col stride is 1.
            // Slicing every 2nd element makes strides: 20 and 2.
            CHECK(dl_tensor.strides[0] == 20);
            CHECK(dl_tensor.strides[1] == 2);

            dl_managed->deleter(dl_managed);
        }

        // Lifetime & Ownership Check
        {
            // We want to ensure the data persists even if the original
            // TorchDataArray dies, provided the DLPack struct is still alive.

            DLManagedTensorVersioned *dl_managed = nullptr;
            void *raw_data_ptr = nullptr;

            {
                // Inner scope: Create array and get DLPack
                auto temp_tensor =
                    torch::ones({5}, torch::TensorOptions().dtype(torch::kF64));
                auto temp_array = TorchDataArray(temp_tensor);

                dl_managed = temp_array.as_dlpack(cpu_device, nullptr, version);
                raw_data_ptr = dl_managed->dl_tensor.data;

                // Sanity check before scope exit
                auto *data = static_cast<double *>(raw_data_ptr);
                CHECK(data[0] == 1.0);
            }
            // temp_array is now destroyed.
            // However, dl_managed should still hold a reference to the storage
            // so data can be read.

            auto *data = static_cast<double *>(raw_data_ptr);
            CHECK(data[0] == 1.0);
            CHECK(data[4] == 1.0);

            dl_managed->deleter(dl_managed);
        }

        // Version Compatibility Checks
        {
            // Major version mismatch
            // If the caller requests a different major version, it should fail.
            DLPackVersion bad_major = {DLPACK_MAJOR_VERSION + 1, DLPACK_MINOR_VERSION};
            CHECK_THROWS_WITH(
                array.as_dlpack(cpu_device, nullptr, bad_major),
                Catch::Matchers::Contains("Caller requested incompatible version")
            );

            // Minor version too old
            // If the caller supports a lower max minor version than the tensor provides,
            // we must ensure it throws (as per logic: max_version.minor < mta_version.minor).
            if (DLPACK_MINOR_VERSION > 0) {
                DLPackVersion old_minor = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION - 1};
                CHECK_THROWS_WITH(
                    array.as_dlpack(cpu_device, nullptr, old_minor),
                    Catch::Matchers::Contains("Caller requested incompatible version")
                );
            }
        }

        // Stream Validation on CPU
        {
            // Passing a stream (non-null) to a CPU tensor should throw an error.
            void* fake_stream = reinterpret_cast<void*>(0x1234);
            CHECK_THROWS_WITH(
                array.as_dlpack(cpu_device, fake_stream, version),
                Catch::Matchers::Contains("Stream must be NULL for CPU tensors")
            );
        }

#ifdef USE_CUDA
        // CUDA Stream Handling (Conditional)
        if (torch::cuda::is_available()) {
            // Create a CUDA tensor
            auto opts = torch::TensorOptions().dtype(torch::kF64).device(torch::kCUDA);
            auto cuda_tensor = torch::rand({10, 10}, opts);
            auto cuda_array = TorchDataArray(cuda_tensor);
            
            DLDevice cuda_device = {kDLCUDA, (int)cuda_tensor.device().index()};

            // Success: passing nullptr stream to CUDA is valid (uses default stream)
            {
                auto dl_managed = cuda_array.as_dlpack(cuda_device, nullptr, version);
                CHECK(dl_managed->dl_tensor.device.device_type == kDLCUDA);
                dl_managed->deleter(dl_managed);
            }

            // Success: passing a valid stream
            {
                c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();
                void* stream_ptr = static_cast<void*>(stream.stream());
                
                auto dl_managed = cuda_array.as_dlpack(cuda_device, stream_ptr, version);
                CHECK(dl_managed->dl_tensor.device.device_type == kDLCUDA);
                dl_managed->deleter(dl_managed);
            }
            
            // Error: Wrong device type requested for CUDA tensor
            {
                CHECK_THROWS_WITH(
                    cuda_array.as_dlpack(cpu_device, nullptr, version),
                    Catch::Matchers::Contains("Requested device does not match")
                );
            }
        }
        if (torch::cuda::is_available()) {
            // Setup a CUDA tensor
            auto opts = torch::TensorOptions().dtype(torch::kF64).device(torch::kCUDA);
            auto cuda_tensor = torch::rand({10, 10}, opts);
            auto cuda_array = TorchDataArray(cuda_tensor);
            
            DLDevice cuda_device = {kDLCUDA, (int)cuda_tensor.device().index()};

            // 1. CUDA -> CUDA (Same device)
            {
                auto dl_managed = cuda_array.as_dlpack(cuda_device, nullptr, version);
                CHECK(dl_managed->dl_tensor.device.device_type == kDLCUDA);
                dl_managed->deleter(dl_managed);
            }

            // 2. CUDA -> CPU (Explicit Transfer)
            // This used to throw, now it should succeed by copying data.
            {
                auto dl_managed = cuda_array.as_dlpack(cpu_device, nullptr, version);
                
                CHECK(dl_managed->dl_tensor.device.device_type == kDLCPU);
                
                // Verify data is readable on CPU
                double* data = static_cast<double*>(dl_managed->dl_tensor.data);
                CHECK(data != nullptr); // Just check pointer validity
                
                dl_managed->deleter(dl_managed);
            }

            // 3. CPU -> CUDA (Explicit Transfer)
            {
                auto dl_managed = array.as_dlpack(cuda_device, nullptr, version);
                CHECK(dl_managed->dl_tensor.device.device_type == kDLCUDA);
                dl_managed->deleter(dl_managed);
            }
        }
#endif
    }
}
