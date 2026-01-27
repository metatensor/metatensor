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
}

TEST_CASE("DLPack conversion") {
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};

    SECTION("basic conversion") {
        auto tensor = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kF64));
        auto array = TorchDataArray(tensor);

        auto* dl_managed = array.as_dlpack(
            /*device=*/{kDLCPU, 0},
            /*stream=*/nullptr,
            /*max_version=*/version
        );

        // Check version
        CHECK(dl_managed->version.major == DLPACK_MAJOR_VERSION);
        CHECK(dl_managed->version.minor <= 3);

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

        // The DLPack pointer should point to the same memory as the Torch
        // tensor
        CHECK(dl_tensor.data == array.tensor().data_ptr());

        dl_managed->deleter(dl_managed);
    }

    SECTION("int32 data type") {
        auto tensor = torch::tensor({2, 3}, torch::TensorOptions().dtype(torch::kInt32));
        auto array = TorchDataArray(tensor);
        auto* dl_managed = array.as_dlpack(
            /*device=*/{kDLCPU, 0},
            /*stream=*/nullptr,
            /*max_version=*/version
        );

        CHECK(dl_managed->dl_tensor.dtype.code == kDLInt);
        CHECK(dl_managed->dl_tensor.dtype.bits == 32);

        dl_managed->deleter(dl_managed);
    }

    SECTION("strides for non contiguous memory") {
        // Create a non-contiguous tensor by slicing
        auto base_tensor = torch::zeros({10, 10}, torch::TensorOptions().dtype(torch::kF64));
        // Take every 2nd element along both dimensions
        auto sliced_tensor = base_tensor.slice(0, 0, 10, 2).slice(1, 0, 10, 2);
        auto sliced_array = TorchDataArray(sliced_tensor);

        CHECK_FALSE(sliced_tensor.is_contiguous());

        auto* dl_managed = sliced_array.as_dlpack(
            /*device=*/{kDLCPU, 0},
            /*stream=*/nullptr,
            /*max_version=*/version
        );
        auto dl_tensor = dl_managed->dl_tensor;

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

    SECTION("lifetime and ownership") {
        // We want to ensure the data persists even if the original
        // TorchDataArray dies, provided the DLPack struct is still alive.
        DLManagedTensorVersioned *dl_managed = nullptr;
        void *raw_data_ptr = nullptr;

        {
            // Inner scope: Create array and get DLPack
            auto temp_tensor =
                torch::ones({5}, torch::TensorOptions().dtype(torch::kF64));
            auto temp_array = TorchDataArray(temp_tensor);

            dl_managed = temp_array.as_dlpack(
                /*device=*/{kDLCPU, 0},
                /*stream=*/nullptr,
                /*max_version=*/version
            );
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

    SECTION("version compatibility") {
        auto tensor = torch::zeros({4}, torch::TensorOptions().dtype(torch::kF64));
        auto array = TorchDataArray(tensor);

        // Major version mismatch
        // If the caller requests a different major version, it should fail.
        DLPackVersion bad_major = {DLPACK_MAJOR_VERSION + 1, DLPACK_MINOR_VERSION};
        CHECK_THROWS_WITH(
            array.as_dlpack(/*device=*/{kDLCPU, 0}, /*stream=*/nullptr, /*max_version=*/ bad_major),
            Catch::Matchers::Contains("Caller requested incompatible version")
        );

        // Minor version too old
        // If the caller supports a lower max minor version than the tensor provides,
        // we must ensure it throws (as per logic: max_version.minor < mta_version.minor).
        if (DLPACK_MINOR_VERSION > 0) {
            DLPackVersion old_minor = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION - 1};
            CHECK_THROWS_WITH(
                array.as_dlpack(/*device=*/{kDLCPU, 0}, /*stream=*/nullptr, /*max_version=*/ old_minor),
                Catch::Matchers::Contains("Caller requested incompatible version")
            );
        }
    }

    SECTION("stream synchronization") {
        auto tensor = torch::zeros({4}, torch::TensorOptions().dtype(torch::kF64));
        auto array = TorchDataArray(tensor);

        // Passing a stream (non-null) to a CPU tensor should throw an error.
        const auto* fake_stream = reinterpret_cast<const int64_t*>(0x1234);
        CHECK_THROWS_WITH(
            array.as_dlpack(/*device=*/{kDLCPU, 0}, /*stream=*/fake_stream, /*max_version=*/version),
            Catch::Matchers::Contains("Stream must be NULL for CPU tensors")
        );

        if (torch::cuda::is_available()) {
            // Create a CUDA tensor
            auto opts = torch::TensorOptions().dtype(torch::kF64).device(torch::kCUDA);
            auto cuda_tensor = torch::rand({10, 10}, opts);
            auto cuda_array = TorchDataArray(cuda_tensor);

            // passing nullptr stream to CUDA is valid (uses default stream)
            auto* dl_managed = cuda_array.as_dlpack(/*device=*/{kDLCUDA, 0}, /*stream=*/nullptr, /*max_version=*/version);
            CHECK(dl_managed->dl_tensor.device.device_type == kDLCUDA);
            dl_managed->deleter(dl_managed);

            // // passing a valid stream
            // auto stream = c10::cuda::getStreamFromPool();
            // CHECK(stream != at::cuda::getDefaultCUDAStream(cuda_tensor.device().index()));

            int64_t stream_idx = 33;
            CHECK_THROWS_WITH(cuda_array.as_dlpack(
                /*device=*/{kDLCUDA, 0},
                /*stream=*/&stream_idx,
                /*max_version=*/version
            ), Catch::Matchers::Contains("CUDA stream support is not implemented yet"));
            // CHECK(dl_managed->dl_tensor.device.device_type == kDLCUDA);
            // dl_managed->deleter(dl_managed);
        }
    }

    SECTION("device transfer") {
        if (torch::cuda::is_available()) {
            auto opts = torch::TensorOptions().dtype(torch::kF64);
            auto cuda_tensor = torch::rand({10}, opts.device(torch::kCUDA));
            auto cuda_array = TorchDataArray(cuda_tensor);

            // CUDA -> CUDA (Same device)
            auto* dl_managed = cuda_array.as_dlpack(/*device=*/{kDLCUDA, 0}, /*stream=*/nullptr, /*max_version=*/version);
            CHECK(dl_managed->dl_tensor.device.device_type == kDLCUDA);
            dl_managed->deleter(dl_managed);

            // CUDA -> CPU (Explicit Transfer)
            dl_managed = cuda_array.as_dlpack(/*device=*/{kDLCPU, 0}, /*stream=*/nullptr, /*max_version=*/version);

            CHECK(dl_managed->dl_tensor.device.device_type == kDLCPU);

            // Verify data is readable on CPU
            auto* data = static_cast<double*>(dl_managed->dl_tensor.data);
            REQUIRE(data != nullptr);

            auto cpu_reference = cuda_tensor.to(torch::kCPU);
            for (int64_t i = 0; i < cpu_reference.numel(); ++i) {
                CHECK(data[i] == cpu_reference[i].item<double>());
            }

            dl_managed->deleter(dl_managed);

            // CPU -> CUDA (Explicit Transfer)
            auto cpu_tensor = torch::rand({10, 10}, opts.device(torch::kCPU));
            auto cpu_array = TorchDataArray(cpu_tensor);

            dl_managed = cpu_array.as_dlpack(/*device=*/{kDLCUDA, 0}, /*stream=*/nullptr, /*max_version=*/version);
            CHECK(dl_managed->dl_tensor.device.device_type == kDLCUDA);
            dl_managed->deleter(dl_managed);
        }
    }
}
