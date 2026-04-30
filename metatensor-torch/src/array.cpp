#include <vector>
#include <cstdint>

#include <torch/script.h>

#include <metatensor.hpp>

#include "metatensor/torch/array.hpp"
#include <ATen/DLConvertor.h>

#include <torch/version.h>

using namespace metatensor_torch;


torch::Device metatensor_torch::dlpack_device_to_torch(DLDevice device) {
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
    case kDLExtDev:
        type = torch::DeviceType::Meta;
        break;
    default:
        throw metatensor::Error(
            "TorchDataArray: Unsupported or unmapped DLPack device type: " +
            std::to_string(device.device_type));
    }

    return torch::Device(type);
}

static DLDevice torch_device_to_dlpack(torch::Device device) {
    DLDeviceType dl_type;
    switch (device.type()) {
    case torch::DeviceType::CPU:
        dl_type = kDLCPU;
        break;
    case torch::DeviceType::CUDA:
        dl_type = kDLCUDA;
        break;
    case torch::DeviceType::HIP:
        dl_type = kDLROCM;
        break;
    case torch::DeviceType::MPS:
        dl_type = kDLMetal;
        break;
    case torch::DeviceType::XPU:
        dl_type = kDLOneAPI;
        break;
    case torch::DeviceType::XLA:
        dl_type = kDLTrn;
        break;
    case torch::DeviceType::Vulkan:
        dl_type = kDLVulkan;
        break;
    case torch::DeviceType::Meta:
        dl_type = kDLExtDev;
        break;
    default:
        throw metatensor::Error(
            "unsupported torch device in torch_device_to_dlpack: " +
            std::string(c10::DeviceTypeName(device.type())));
    }

    return DLDevice{dl_type, static_cast<int32_t>(device.index() < 0 ? 0 : device.index())};
}

static DLDataType torch_dtype_to_dlpack(torch::ScalarType dtype) {
    DLDataType dt;
    dt.lanes = 1;

    switch (dtype) {
    case torch::kFloat16:
        dt.code = kDLFloat;
        dt.bits = 16;
        break;
    case torch::kFloat32:
        dt.code = kDLFloat;
        dt.bits = 32;
        break;
    case torch::kFloat64:
        dt.code = kDLFloat;
        dt.bits = 64;
        break;
    case torch::kBFloat16:
        dt.code = kDLBfloat;
        dt.bits = 16;
        break;
    case torch::kInt8:
        dt.code = kDLInt;
        dt.bits = 8;
        break;
    case torch::kInt16:
        dt.code = kDLInt;
        dt.bits = 16;
        break;
    case torch::kInt32:
        dt.code = kDLInt;
        dt.bits = 32;
        break;
    case torch::kInt64:
        dt.code = kDLInt;
        dt.bits = 64;
        break;
    case torch::kUInt8:
        dt.code = kDLUInt;
        dt.bits = 8;
        break;
    case torch::kBool:
        dt.code = kDLBool;
        dt.bits = 8;
        break;
    case torch::kComplexHalf:
        dt.code = kDLComplex;
        dt.bits = 32;
        break;
    case torch::kComplexFloat:
        dt.code = kDLComplex;
        dt.bits = 64;
        break;
    case torch::kComplexDouble:
        dt.code = kDLComplex;
        dt.bits = 128;
        break;
    default:
        throw metatensor::Error(
            "unknown or unsupported torch dtype in torch_dtype_to_dlpack: " +
            std::string(c10::toString(dtype))
        );
    }

    return dt;
}


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

std::unique_ptr<metatensor::DataArrayBase> TorchDataArray::copy(DLDevice device) const {
    auto torch_device = dlpack_device_to_torch(device);
    auto tensor_copy = this->tensor().to(torch_device, /*non_blocking=*/true, /*copy=*/true);
    return std::unique_ptr<DataArrayBase>(new TorchDataArray(std::move(tensor_copy)));
}

std::unique_ptr<metatensor::DataArrayBase> TorchDataArray::create(
    std::vector<uintptr_t> shape,
    metatensor::MtsArray fill_value
) const {
    auto sizes = std::vector<int64_t>();
    for (auto size: shape) {
        sizes.push_back(static_cast<int64_t>(size));
    }

    DLDevice cpu_device = {kDLCPU, 0};
    DLPackVersion max_version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto* fill_value_dlpack = fill_value.as_dlpack(cpu_device, nullptr, max_version);

    // Validate fill_value shape from the DLPack tensor directly
    if (fill_value_dlpack->dl_tensor.ndim != 0) {
        if (fill_value_dlpack->deleter != nullptr) {
            fill_value_dlpack->deleter(fill_value_dlpack);
        }
        throw std::runtime_error("`fill_value` must be a single scalar");
    }

    c10::Scalar scalar_val;
    auto code = fill_value_dlpack->dl_tensor.dtype.code;
    auto bits = fill_value_dlpack->dl_tensor.dtype.bits;
    assert(fill_value_dlpack->dl_tensor.dtype.lanes == 1);

    auto expected_dtype = this->dtype();
    if (code != expected_dtype.code || bits != expected_dtype.bits) {
        if (fill_value_dlpack->deleter != nullptr) {
            fill_value_dlpack->deleter(fill_value_dlpack);
        }
        throw std::runtime_error(
            "`fill_value` dtype (" + std::to_string(code) + " with " + std::to_string(bits) + " bits) does not match the array dtype (" +
            std::to_string(expected_dtype.code) + " with " + std::to_string(expected_dtype.bits) + " bits)"
        );
    }

    // Account for DLPack byte_offset per spec
    const auto* data = static_cast<const char*>(fill_value_dlpack->dl_tensor.data);
    data += fill_value_dlpack->dl_tensor.byte_offset;

    // TODO: switch to at::fromDLPackVersioned() when our MSTorchV is 2.9
    if (code == kDLFloat && bits == 64) {
        scalar_val = *reinterpret_cast<const double*>(data);
    } else if (code == kDLFloat && bits == 32) {
        scalar_val = *reinterpret_cast<const float*>(data);
    } else if (code == kDLFloat && bits == 16) {
        scalar_val = *reinterpret_cast<const at::Half*>(data);
    } else if (code == kDLBfloat && bits == 16) {
        scalar_val = *reinterpret_cast<const at::BFloat16*>(data);
    } else if (code == kDLInt && bits == 64) {
        scalar_val = *reinterpret_cast<const int64_t*>(data);
    } else if (code == kDLInt && bits == 32) {
        scalar_val = *reinterpret_cast<const int32_t*>(data);
    } else if (code == kDLInt && bits == 16) {
        scalar_val = *reinterpret_cast<const int16_t*>(data);
    } else if (code == kDLInt && bits == 8) {
        scalar_val = *reinterpret_cast<const int8_t*>(data);
    } else if (code == kDLUInt && bits == 8) {
        scalar_val = *reinterpret_cast<const uint8_t*>(data);
    } else if (code == kDLBool && bits == 8) {
        scalar_val = *reinterpret_cast<const bool*>(data);
    } else if (code == kDLComplex && bits == 128) {
        auto real = *reinterpret_cast<const double*>(data);
        auto imag = *reinterpret_cast<const double*>(data + sizeof(double));
        scalar_val = c10::complex<double>(real, imag);
    } else if (code == kDLComplex && bits == 64) {
        auto real = *reinterpret_cast<const float*>(data);
        auto imag = *reinterpret_cast<const float*>(data + sizeof(float));
        scalar_val = c10::complex<float>(real, imag);
    } else if (code == kDLComplex && bits == 32) {
        auto real = *reinterpret_cast<const at::Half*>(data);
        auto imag = *reinterpret_cast<const at::Half*>(data + sizeof(at::Half));
        scalar_val = c10::complex<at::Half>(real, imag);
    } else {
        if (fill_value_dlpack->deleter != nullptr) {
            fill_value_dlpack->deleter(fill_value_dlpack);
        }
        throw std::runtime_error(
            "unsupported fill_value DLPack dtype: code " + std::to_string(code) +
            " with " + std::to_string(bits) + " bits"
        );
    }
    if (fill_value_dlpack->deleter != nullptr) {
        fill_value_dlpack->deleter(fill_value_dlpack);
    }

    return std::unique_ptr<DataArrayBase>(new TorchDataArray(
        torch::full(
            sizes,
            scalar_val,
            torch::TensorOptions()
                .dtype(this->tensor().dtype())
                .device(this->tensor().device())
        )
    ));
}

// Wraps legacy DLManagedTensor in DLManagedTensorVersioned. Required because
// at::toDLPack() returns the legacy format.
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

DLDevice TorchDataArray::device() const {
    if (!tensor_.has_storage()) {
        // use the same device as "meta", since we can not access the
        // corresponding data
        return DLDevice{kDLExtDev, 0};
    }
    return torch_device_to_dlpack(tensor_.device());
}

DLDataType TorchDataArray::dtype() const {
    return torch_dtype_to_dlpack(tensor_.scalar_type());
}

DLManagedTensorVersioned* TorchDataArray::as_dlpack(DLDevice device, const int64_t* stream, DLPackVersion max_version) {
    DLPackVersion mta_version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    bool major_mismatch = max_version.major != mta_version.major;
    bool minor_too_high = max_version.minor < mta_version.minor;
    if (major_mismatch || minor_too_high) {
        throw metatensor::Error(
            "TorchDataArray supports DLPack version " + std::to_string(mta_version.major) + "." +
            std::to_string(mta_version.minor) + ". Caller requested incompatible version " +
            std::to_string(max_version.major) + "." + std::to_string(max_version.minor)
        );
    }
    torch::Device target_device = dlpack_device_to_torch(device);
    torch::Tensor tensor_to_pack = this->tensor_;

    if (!tensor_to_pack.has_storage()) {
        // For tensors without storage, use the same code path as meta tensors
        tensor_to_pack = tensor_to_pack.to(torch::kMeta);
    }

    if (tensor_to_pack.device() != target_device) {
        tensor_to_pack = tensor_to_pack.to(target_device, /*non_blocking=*/false);
    }

    // Stream sync ala:
    // https://github.com/pytorch/pytorch/blob/eb65b36914d039f37e24c2e0372f9e7c022f20ed/torch/_tensor.py#L1784-L1819
    if (tensor_to_pack.is_cuda()) {
        if (stream != nullptr && *stream != 0) {
            throw metatensor::Error(
                "TorchDataArray: CUDA stream synchronization is not yet "
                "implemented. The required functions (at::cuda::getCurrentCUDAStream, "
                "at::cuda::getStreamFromExternal) are in libtorch_cuda.so, which "
                "cannot be linked in pre-built CPU+CUDA wheels. Pass stream=nullptr "
                "for default stream behavior."
            );
        }

        // Stream synchronization requires libtorch_cuda.so. Linking it
        // unconditionally breaks pre-built wheels that must work in both
        // CUDA and CPU-only environments.

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

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9
    return at::toDLPackVersioned(tensor_to_pack);

#elif TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR <= 8
    // at::toDLPackVersioned() is available starting in 2.9, so we use the
    // legacy at::toDLPack() and wrap it in a versioned struct below.
    DLManagedTensor* legacy_tensor = at::toDLPack(tensor_to_pack);

    auto* versioned_tensor = new DLManagedTensorVersioned();
    versioned_tensor->version = mta_version;
    // Setup context, keeping the legacy variant for the deleter
    versioned_tensor->manager_ctx = legacy_tensor;
    versioned_tensor->deleter = dlpack_versioned_deleter;
    versioned_tensor->flags = 0;
    // Copy metadata
    versioned_tensor->dl_tensor = legacy_tensor->dl_tensor;
    return versioned_tensor;
#else
    throw metatensor::Error(
        "unsupported PyTorch version for DLPack export: " +
        std::to_string(TORCH_VERSION_MAJOR) + "." + std::to_string(TORCH_VERSION_MINOR)
    );
#endif
}

/// Use a full DLManagedTensorVersioned as the context for the legacy deleter
/// This is used for PyTorch <=2.8, which does not have at::toDLPackVersioned()
struct ManagedTensorCtx {
    DLManagedTensorVersioned* versioned_tensor;
};

std::unique_ptr<metatensor::DataArrayBase> TorchDataArray::from_dlpack(DLManagedTensorVersioned *dl_tensor) const {
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9
    auto tensor = at::fromDLPackVersioned(dl_tensor);
#elif TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR <= 8
    // create a DLManagedTensor that wraps a DLManagedTensorVersioned, so we
    // can use at::fromDLPack() for PyTorch <=2.8
    auto* managed_tensor = new DLManagedTensor();
    managed_tensor->dl_tensor = dl_tensor->dl_tensor;
    managed_tensor->manager_ctx = new ManagedTensorCtx{dl_tensor};
    managed_tensor->deleter = [](DLManagedTensor* self) {
        if (self != nullptr) {
            auto* ctx = static_cast<ManagedTensorCtx*>(self->manager_ctx);
            if (ctx != nullptr && ctx->versioned_tensor != nullptr && ctx->versioned_tensor->deleter != nullptr) {
                ctx->versioned_tensor->deleter(ctx->versioned_tensor);
            }
            delete ctx;
            delete self;
        }
    };

    auto tensor = at::fromDLPack(managed_tensor);
#else
    throw metatensor::Error(
        "unsupported PyTorch version for DLPack import: " +
        std::to_string(TORCH_VERSION_MAJOR) + "." + std::to_string(TORCH_VERSION_MINOR)
    );
#endif

    return std::unique_ptr<metatensor::DataArrayBase>(new TorchDataArray(std::move(tensor)));
}

const std::vector<uintptr_t>& TorchDataArray::shape() const & {
    return shape_;
}

void TorchDataArray::reshape(const std::vector<uintptr_t>& shape) {
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

void TorchDataArray::move_data(
    const metatensor::DataArrayBase& raw_input,
    std::vector<mts_data_movement_t> moves
) {
    using torch::indexing::Slice;
    using torch::indexing::Ellipsis;

    const auto& input = dynamic_cast<const TorchDataArray&>(raw_input);
    auto input_tensor = input.tensor();
    auto output_tensor = this->tensor();

    assert(input_tensor.dtype() == output_tensor.dtype());
    assert(input_tensor.device() == output_tensor.device());

    if (input_tensor.device() == torch::kMeta) {
        // tensors on the "meta" device contain no data to move around,
        // and the code below crashes for some old PyTorch versions
        return;
    }

    if (moves.empty()) {
        return;
    }

    // Check if we can use the optimized path (all moves have same property structure)
    bool constant_properties = true;
    auto first_prop_start_in = moves[0].properties_start_in;
    auto first_prop_start_out = moves[0].properties_start_out;
    auto first_prop_len = moves[0].properties_length;

    for (const auto& move : moves) {
        if (move.properties_start_in != first_prop_start_in ||
            move.properties_start_out != first_prop_start_out ||
            move.properties_length != first_prop_len) {
            constant_properties = false;
            break;
        }
    }

    if (constant_properties) {
        auto sample_in_indices = std::vector<int64_t>();
        auto sample_out_indices = std::vector<int64_t>();
        sample_in_indices.reserve(moves.size());
        sample_out_indices.reserve(moves.size());

        bool contiguous_in = true;
        bool contiguous_out = true;

        if (moves.size() > 1) {
            for (size_t i = 1; i < moves.size(); ++i) {
                if (moves[i].sample_in != moves[i - 1].sample_in + 1) {
                    contiguous_in = false;
                }
                if (moves[i].sample_out != moves[i - 1].sample_out + 1) {
                    contiguous_out = false;
                }
            }
        }

        for (const auto& move : moves) {
            sample_in_indices.push_back(static_cast<int64_t>(move.sample_in));
            sample_out_indices.push_back(static_cast<int64_t>(move.sample_out));
        }

        torch::Tensor samples_in;
        torch::Tensor samples_out;

        auto property_start_in = static_cast<int64_t>(first_prop_start_in);
        auto property_start_out = static_cast<int64_t>(first_prop_start_out);
        auto property_len = static_cast<int64_t>(first_prop_len);

        torch::Tensor input_slice;
        if (contiguous_in) {
            auto start = static_cast<int64_t>(moves[0].sample_in);
            auto end = start + static_cast<int64_t>(moves.size());
            input_slice = input_tensor.index({
                Slice(start, end),
                Ellipsis,
                Slice(property_start_in, property_start_in + property_len)
            });
        } else {
            samples_in = torch::from_blob(
                sample_in_indices.data(),
                {static_cast<int64_t>(sample_in_indices.size())},
                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)
            ).to(input_tensor.device(), /*non_blocking=*/true, /*copy=*/true);

            input_slice = input_tensor.index({
                samples_in,
                Ellipsis,
                Slice(property_start_in, property_start_in + property_len)
            });
        }

        if (contiguous_out) {
            auto samples_start = static_cast<int64_t>(moves[0].sample_out);
            auto samples_end = samples_start + static_cast<int64_t>(moves.size());
            output_tensor.index_put_(
                {
                    Slice(samples_start, samples_end),
                    Ellipsis,
                    Slice(property_start_out, property_start_out + property_len)
                },
                input_slice
            );
        } else {
            samples_out = torch::from_blob(
                sample_out_indices.data(),
                {static_cast<int64_t>(sample_out_indices.size())},
                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)
            ).to(input_tensor.device(), /*non_blocking=*/true, /*copy=*/true);

            output_tensor.index_put_(
                {
                    samples_out,
                    Ellipsis,
                    Slice(property_start_out, property_start_out + property_len)
                },
                input_slice
            );
        }
    } else {
        for (const auto& move : moves) {
            auto sample_in = static_cast<int64_t>(move.sample_in);
            auto sample_out = static_cast<int64_t>(move.sample_out);

            auto property_start_in = static_cast<int64_t>(move.properties_start_in);
            auto property_start_out = static_cast<int64_t>(move.properties_start_out);
            auto property_len = static_cast<int64_t>(move.properties_length);

            auto input_slice = input_tensor.index({
                sample_in,
                Ellipsis,
                Slice(property_start_in, property_start_in + property_len)
            });

            output_tensor.index_put_(
                {
                    sample_out,
                    Ellipsis,
                    Slice(property_start_out, property_start_out + property_len)
                },
                input_slice
            );
        }
    }
}

void TorchDataArray::update_shape() {
    shape_.clear();
    for (auto size: this->tensor_.sizes()) {
        shape_.push_back(static_cast<uintptr_t>(size));
    }
}
