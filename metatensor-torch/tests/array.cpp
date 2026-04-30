#include <torch/torch.h>

#include <metatensor/torch.hpp>
using namespace metatensor_torch;

#include <catch.hpp>

static std::vector<std::pair<torch::Device, DLDevice>> available_devices() {
    auto devices = std::vector<std::pair<torch::Device, DLDevice>>();

    devices.emplace_back(torch::kCPU, DLDevice{kDLCPU, 0});

    if (torch::cuda::is_available()) {
        devices.emplace_back(torch::Device(torch::kCUDA, 0), DLDevice{kDLCUDA, 0});
    }

    if (torch::mps::is_available()) {
        devices.emplace_back(torch::Device(torch::kMPS, 0), DLDevice{kDLMetal, 0});
    }

    return devices;
}

TEST_CASE("Arrays") {
    auto tensor = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kF32));
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
        auto copy = array.copy({kDLCPU, 0});
        auto* copy_ptr = dynamic_cast<TorchDataArray*>(copy.get());

        CHECK(copy_ptr->tensor().data_ptr() != array.tensor().data_ptr());
        CHECK(copy_ptr->tensor().sizes() == std::vector<int64_t>{2, 3, 4});
        CHECK(copy_ptr->tensor().dtype() == torch::kF32);


        // make a copy on device
        for (auto [device, dl_device]: available_devices()) {
            auto device_copy = array.copy(dl_device);
            auto* device_copy_ptr = dynamic_cast<TorchDataArray*>(device_copy.get());

            CHECK(device_copy_ptr->tensor().sizes() == std::vector<int64_t>{2, 3, 4});
            CHECK(device_copy_ptr->tensor().dtype() == torch::kF32);
            CHECK(device_copy_ptr->tensor().device() == device);
        }

        auto fill_value = metatensor::DataArrayBase::to_mts_array(
            std::make_unique<TorchDataArray>(torch::zeros({}, torch::kF32))
        );
        auto created = array.create({5, 6}, std::move(fill_value));
        auto* created_ptr = dynamic_cast<TorchDataArray*>(created.get());

        CHECK((created_ptr->tensor().sizes() == std::vector<int64_t>{5, 6}));
        CHECK(created_ptr->tensor().dtype() == torch::kF32);
    }

    SECTION("create with any dtype") {
        auto ALL_DTYPES = std::array<torch::ScalarType, 14>{
            torch::kBool,
            torch::kF64,
            torch::kF32,
            torch::kF16,
            torch::kBFloat16,
            torch::kInt64,
            torch::kInt32,
            torch::kInt16,
            torch::kInt8,
            torch::kUInt8,
            torch::kBool,
            torch::kComplexDouble,
            torch::kComplexFloat,
            torch::kComplexHalf,
        };
        for (auto dtype: ALL_DTYPES) {
            auto options = torch::TensorOptions().dtype(dtype);
            auto dtype_array = TorchDataArray(torch::zeros({1, 2}, options));

            torch::Tensor fill_value;
            if (dtype == torch::kBool) {
                fill_value = torch::ones({}, options);
            } else if (
                dtype == torch::kInt64 || dtype == torch::kInt32 || dtype == torch::kInt16 ||
                dtype == torch::kInt8 || dtype == torch::kUInt64 || dtype == torch::kUInt32 ||
                dtype == torch::kUInt16 || dtype == torch::kUInt8
            ) {
                fill_value = torch::randint(0, 42, {}, options);
            } else {
                fill_value = torch::rand({}, options);
            }

            auto fill_value_mts = metatensor::DataArrayBase::to_mts_array(
                std::make_unique<TorchDataArray>(fill_value)
            );

            auto created = dtype_array.create({3, 4}, fill_value_mts);
            auto* created_ptr = dynamic_cast<TorchDataArray*>(created.get());

            CHECK(created_ptr->tensor().dtype() == dtype);
            CHECK(torch::all(created_ptr->tensor() == torch::full({3, 4}, fill_value.item(), options)).item<bool>());
        }
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
            ), Catch::Matchers::Contains("CUDA stream synchronization is not yet implemented"));
            // CHECK(dl_managed->dl_tensor.device.device_type == kDLCUDA);
            // dl_managed->deleter(dl_managed);
        }
    }

    SECTION("device transfer") {
        for (auto [device, dl_device]: available_devices()) {
            if (device.is_mps() && TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 8) {
                // MPS DLPack support was added in PyTorch 2.8
                continue;
            }

            auto opts = torch::TensorOptions().dtype(torch::kF32);
            auto cuda_tensor = torch::rand({10}, opts.device(device));
            auto cuda_array = TorchDataArray(cuda_tensor);

            // transfer to the same device
            auto* dl_managed = cuda_array.as_dlpack(/*device=*/dl_device, /*stream=*/nullptr, /*max_version=*/version);
            CHECK(dl_managed->dl_tensor.device.device_type == dl_device.device_type);
            dl_managed->deleter(dl_managed);

            // transfer to CPU
            dl_managed = cuda_array.as_dlpack(/*device=*/{kDLCPU, 0}, /*stream=*/nullptr, /*max_version=*/version);

            CHECK(dl_managed->dl_tensor.device.device_type == kDLCPU);

            // Verify data is readable on CPU
            auto* data = static_cast<float*>(dl_managed->dl_tensor.data);
            REQUIRE(data != nullptr);

            auto cpu_reference = cuda_tensor.to(torch::kCPU);
            for (int64_t i = 0; i < cpu_reference.numel(); ++i) {
                CHECK(data[i] == cpu_reference[i].item<float>());
            }

            dl_managed->deleter(dl_managed);

            // cpu to device
            auto cpu_tensor = torch::rand({10, 10}, opts.device(torch::kCPU));
            auto cpu_array = TorchDataArray(cpu_tensor);

            dl_managed = cpu_array.as_dlpack(/*device=*/dl_device, /*stream=*/nullptr, /*max_version=*/version);
            CHECK(dl_managed->dl_tensor.device.device_type == dl_device.device_type);
            dl_managed->deleter(dl_managed);
        }
    }

    SECTION("create from dlpack") {
        auto float_tensor = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kF32));
        auto float_array = TorchDataArray(float_tensor);

        auto int_tensor = torch::ones({4, 7, 8, 1}, torch::TensorOptions().dtype(torch::kI32));
        auto int_array = TorchDataArray(int_tensor);

        auto* float_dlpack = float_array.as_dlpack(
            /*device=*/{kDLCPU, 0},
            /*stream=*/nullptr,
            /*max_version=*/version
        );

        auto* int_dlpack = int_array.as_dlpack(
            /*device=*/{kDLCPU, 0},
            /*stream=*/nullptr,
            /*max_version=*/version
        );

        auto new_float_array = float_array.from_dlpack(float_dlpack);
        // use a different source array dtype
        auto new_int_array = float_array.from_dlpack(int_dlpack);

        auto new_float_tensor = dynamic_cast<TorchDataArray*>(new_float_array.get())->tensor();
        auto new_int_tensor = dynamic_cast<TorchDataArray*>(new_int_array.get())->tensor();

        CHECK(torch::all(new_float_tensor == float_tensor).item<bool>());
        CHECK(torch::all(new_int_tensor == int_tensor).item<bool>());

        // test SimpleDataArray::from_dlpack with a non-contiguous tensor created by torch
        auto base_tensor = torch::zeros({10, 10}, torch::TensorOptions().dtype(torch::kF64));
        auto sliced_tensor = base_tensor.slice(0, 0, 10, 2).slice(1, 0, 10, 2).t();
        CHECK(!sliced_tensor.is_contiguous());
        CHECK(sliced_tensor.strides() == std::vector<int64_t>{2, 20});

        auto sliced_array = TorchDataArray(sliced_tensor);
        auto* sliced_dlpack = sliced_array.as_dlpack(
            /*device=*/{kDLCPU, 0},
            /*stream=*/nullptr,
            /*max_version=*/version
        );

        auto cxx_array = metatensor::DataArrayBase::to_mts_array(
            std::make_unique<metatensor::SimpleDataArray<int8_t>>(metatensor::SimpleDataArray<int8_t>({}))
        );

        auto new_cxx_array = cxx_array.from_dlpack(sliced_dlpack);

        CHECK(new_cxx_array.origin() == cxx_array.origin());
        CHECK(new_cxx_array.origin() != float_array.origin());

        auto* cxx_dlpack = cxx_array.as_dlpack({kDLCPU, 0}, nullptr, version);

        auto tensor_from_cxx = float_array.from_dlpack(cxx_dlpack);

        auto recovered_tensor = dynamic_cast<TorchDataArray*>(tensor_from_cxx.get())->tensor();
        CHECK(torch::all(recovered_tensor == sliced_tensor).item<bool>());
    }
}
