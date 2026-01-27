#include <vector>
#include <cstdint>

#include <torch/script.h>

#include <metatensor.hpp>

#include "metatensor/torch/array.hpp"
#include <ATen/DLConvertor.h>

using namespace metatensor_torch;

// We need to register a data origin with metatensor, that will be used for all
// mts_array_t containing a C++ torch tensor. This is initialized to 0 (meaning
// "no origin"), and will be set by `MetatensorOriginRegistration` in
// `TorchDataArray::origin`.
mts_data_origin_t metatensor_torch::TORCH_DATA_ORIGIN = 0;

struct MetatensorOriginRegistration {
    MetatensorOriginRegistration(const char* name) {
        auto status = mts_register_data_origin(name, &TORCH_DATA_ORIGIN);
        if (status != MTS_SUCCESS) {
            C10_THROW_ERROR(ValueError, "failed to register torch data origin");
        }
    }
};


TorchDataArray::TorchDataArray(torch::Tensor tensor): tensor_(std::move(tensor)) {
    this->update_shape();
}

mts_data_origin_t TorchDataArray::origin() const {
    // mts_data_origin registration in a thread-safe way through C++11 static
    // initialization of a class with a constructor.
    static auto REGISTRATION = MetatensorOriginRegistration("metatensor_torch::TorchDataArray");
    return TORCH_DATA_ORIGIN;
}

std::unique_ptr<metatensor::DataArrayBase> TorchDataArray::copy() const {
    return std::unique_ptr<DataArrayBase>(new TorchDataArray(this->tensor().clone()));
}

std::unique_ptr<metatensor::DataArrayBase> TorchDataArray::create(std::vector<uintptr_t> shape) const {
    auto sizes = std::vector<int64_t>();
    for (auto size: shape) {
        sizes.push_back(static_cast<int64_t>(size));
    }

    return std::unique_ptr<DataArrayBase>(new TorchDataArray(
        torch::zeros(
            sizes,
            torch::TensorOptions()
                .dtype(this->tensor().dtype())
                .device(this->tensor().device())
        )
    ));
}

double* TorchDataArray::data() & {
    if (!this->tensor_.device().is_cpu()) {
        C10_THROW_ERROR(ValueError, "can not access the data of a torch::Tensor not on CPU");
    }

    if (this->tensor_.dtype() != torch::kF64) {
        C10_THROW_ERROR(ValueError,
            "can not access the data of this torch::Tensor: expected a dtype "
            "of float64, got " + std::string(this->tensor_.dtype().name())
        );
    }

    if (!this->tensor_.is_contiguous()) {
        C10_THROW_ERROR(ValueError, "can not access the data of a non contiguous torch::Tensor");
    }

    return static_cast<double*>(this->tensor_.data_ptr());
}

// Helpful deleter: destroys legacy DLPack tensor when the versioned one is done
// TODO(rg): shouldn't be required later, we we shift to at::toDLPackVersioned()
static void dlpack_versioned_deleter(DLManagedTensorVersioned* self) {
    if (self != nullptr) {
        // Retrieve the legacy tensor stored in the context
        auto* legacy_tensor = static_cast<DLManagedTensor*>(self->manager_ctx);
        // Use the legacy deleter to free the internal ATen resources
        if (legacy_tensor != nullptr && legacy_tensor->deleter != nullptr) {
            legacy_tensor->deleter(legacy_tensor);
        }
        // Free the versioned wrapper
        delete self;
    }
}

namespace {

torch::Device dlpack_device_to_torch(DLDevice device) {
    torch::DeviceType type;
    // Reference:
    // https://github.com/pytorch/pytorch/blob/3eddf049221fc04c2ac9d4af53c00305484ef325/c10/core/Device.cpp#L13-L38

    switch (device.device_type) {
    case kDLCPU:
        type = torch::DeviceType::CPU;
        break;
    case kDLCUDA:
        type = torch::DeviceType::CUDA;
        break;
    case kDLCUDAHost:
        // PyTorch treats pinned CUDA memory as CPU-accessible.
        type = torch::DeviceType::CPU;
        break;
    case kDLCUDAManaged:
        type = torch::DeviceType::CUDA;
        break;
    case kDLROCM:
        type = torch::DeviceType::HIP;
        break;
    case kDLROCMHost:
        // PyTorch treats pinned ROCm memory as CPU-accessible.
        type = torch::DeviceType::CPU;
        break;
    case kDLMetal:
        type = torch::DeviceType::MPS;
        break;
    case kDLOneAPI:
        type = torch::DeviceType::XPU;
        break;
    case kDLTrn:
        type = torch::DeviceType::XLA;
        break;
    case kDLVulkan:
        type = torch::DeviceType::Vulkan;
        break;
    default:
        throw metatensor::Error(
            "TorchDataArray: Unsupported or unmapped DLPack device type: " +
            std::to_string(device.device_type));
    }

    return torch::Device(type);
}

} // namespace

DLManagedTensorVersioned* TorchDataArray::as_dlpack(DLDevice device, const int64_t* stream, DLPackVersion max_version) {
    // Uses the existing ATen API to get a legacy DLManagedTensor.
    // TODO(rg): this should eventually just be
    // return at::toDLPackVersioned(this->tensor_);
    // https://github.com/pytorch/pytorch/blob/2.9.1/aten/src/ATen/DLConvertor.cpp
    // ... until then.
    DLPackVersion mta_version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    bool major_mismatch = max_version.major != mta_version.major;
    bool minor_too_high = max_version.minor < mta_version.minor;
    if (major_mismatch || minor_too_high) {
        throw metatensor::Error("TorchDataArray supports DLPack version " +
                                std::to_string(mta_version.major) + "." +
                                std::to_string(mta_version.minor) +
                                ". Caller requested incompatible version " +
                                std::to_string(max_version.major) + "." +
                                std::to_string(max_version.minor));
    }
    torch::Device target_device = dlpack_device_to_torch(device);
    torch::Tensor tensor_to_pack = this->tensor_;

    if (tensor_to_pack.device() != target_device) {
        // consumers should handle synchronization via the stream, but this is
        // the default argument.
        tensor_to_pack = tensor_to_pack.to(target_device, /*non_blocking=*/false);
    }

    // Stream sync ala:
    // https://github.com/pytorch/pytorch/blob/eb65b36914d039f37e24c2e0372f9e7c022f20ed/torch/_tensor.py#L1784-L1819
    if (tensor_to_pack.is_cuda()) {
        if (stream != nullptr && *stream != 0) {
            throw metatensor::Error("TorchDataArray: CUDA stream support is not implemented yet");
        }

        // No implementation for now, since the functions we need are in
        // libtorch_cuda.so, which makes linking complicated for pre-built wheels.

        // auto device = tensor_to_pack.device();

        // auto current_stream = at::cuda::getCurrentCUDAStream(device.index());
        // auto target_stream = at::cuda::getCurrentCUDAStream(device.index());

        // if (stream != nullptr) {
        //     auto c10_stream = c10::Stream(c10::Stream::UNSAFE, device, c10::StreamId(*stream));
        //     target_stream = at::cuda::getStreamFromExternal(
        //         at::cuda::CUDAStream(c10_stream),
        //         device.index()
        //     );
        // }

        // if (current_stream != target_stream) {
        //     // Record event on current stream
        //     auto event = c10::Event(c10::DeviceType::CUDA);
        //     event.record(current_stream);

        //     // and make the target stream wait for it
        //     event.block(target_stream);
        // }
    } else if (tensor_to_pack.is_cpu() && stream != nullptr) {
        throw metatensor::Error(
            "TorchDataArray: Stream must be NULL for CPU tensors");
    } else {
        // ignore stream for all other devices for now
    }

    // Legacy dlpack interface for maximal Torch compatibility
    DLManagedTensor* legacy_tensor = at::toDLPack(tensor_to_pack);
    // Compare the device
    if (legacy_tensor->dl_tensor.device.device_type != device.device_type ||
        legacy_tensor->dl_tensor.device.device_id != device.device_id) {

        // Cleanup the legacy tensor we just created before throwing
        if (legacy_tensor->deleter != nullptr) {
            legacy_tensor->deleter(legacy_tensor);
        }

        throw metatensor::Error(
            "TorchDataArray: Requested device does not match tensor device");
    }
    // Wrap into a versioned struct
    auto* versioned_tensor = new DLManagedTensorVersioned();
    versioned_tensor->version = mta_version;
    // Setup context, keeping the legacy variant for the deleter
    versioned_tensor->manager_ctx = legacy_tensor;
    versioned_tensor->deleter = dlpack_versioned_deleter;
    versioned_tensor->flags = 0;
    // Copy metadata
    versioned_tensor->dl_tensor = legacy_tensor->dl_tensor;
    return versioned_tensor;
}

const std::vector<uintptr_t>& TorchDataArray::shape() const & {
    return shape_;
}

void TorchDataArray::reshape(std::vector<uintptr_t> shape) {
    auto sizes = std::vector<int64_t>();
    for (auto size: shape) {
        sizes.push_back(static_cast<int64_t>(size));
    }

    this->tensor_ = this->tensor().reshape(sizes).contiguous();

    this->update_shape();
}

void TorchDataArray::swap_axes(uintptr_t axis_1, uintptr_t axis_2) {
    this->tensor_ = this->tensor().swapaxes(
        static_cast<int64_t>(axis_1),
        static_cast<int64_t>(axis_2)
    ).contiguous();

    this->update_shape();
}

void TorchDataArray::move_samples_from(
    const metatensor::DataArrayBase& raw_input,
    std::vector<mts_sample_mapping_t> samples,
    uintptr_t property_start,
    uintptr_t property_end
) {
    const auto& input = dynamic_cast<const TorchDataArray&>(raw_input);
    auto input_tensor = input.tensor();

    auto options = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt64);
    auto input_samples = torch::zeros({static_cast<int64_t>(samples.size())}, options);
    auto output_samples = torch::zeros({static_cast<int64_t>(samples.size())}, options);

    for (int64_t i=0; i<samples.size(); i++) {
        input_samples[i] = static_cast<int64_t>(samples[i].input);
        output_samples[i] = static_cast<int64_t>(samples[i].output);
    }

    using torch::indexing::Slice;
    using torch::indexing::Ellipsis;
    auto output_tensor = this->tensor();

    assert(input_tensor.dtype() == output_tensor.dtype());
    assert(input_tensor.device() == output_tensor.device());

    if (input_tensor.device() == torch::kMeta) {
        // tensors on the "meta" device contain no data to move around,
        // and the code below crashes for some old PyTorch versions
        return;
    }

    // output[output_samples, ..., properties] = input[input_samples, ..., :]
    output_tensor.index_put_(
        {output_samples, Ellipsis, Slice(property_start, property_end)},
        input_tensor.index({input_samples, Ellipsis, Slice()})
    );
}

void TorchDataArray::update_shape() {
    shape_.clear();
    for (auto size: this->tensor_.sizes()) {
        shape_.push_back(static_cast<uintptr_t>(size));
    }
}
