#include <vector>
#include <cstdint>

#include <torch/script.h>

#include <metatensor.hpp>

#include "metatensor/torch/array.hpp"

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

DLManagedTensorVersioned* metatensor_torch::TorchDataArray::as_dlpack() const {
    auto* managed = new DLManagedTensorVersioned();
    // Set up version info
    managed->version.major = 1;
    managed->version.minor = 1;
    // TODO(rg): can this just increase the ref count
    auto* tensor_ref = new at::Tensor(tensor_);
    managed->manager_ctx = tensor_ref;
    // Set up the DLTensor
    auto& dl_tensor = managed->dl_tensor;
    // Set device information
    if (tensor_.is_cuda()) {
        dl_tensor.device.device_type = DLDeviceType::kDLCUDA;
        dl_tensor.device.device_id = tensor_.get_device();
    } else {
        dl_tensor.device.device_type = DLDeviceType::kDLCPU;
        dl_tensor.device.device_id = 0;
    }
    // Set tensor dimensions
    dl_tensor.ndim = tensor_.dim();
    // Create shape array
    int64_t* shape = new int64_t[dl_tensor.ndim];
    for (int i = 0; i < dl_tensor.ndim; i++) {
        shape[i] = tensor_.size(i);
    }
    dl_tensor.shape = shape;
    // Create strides array if tensor is not contiguous
    if (tensor_.is_contiguous()) {
        dl_tensor.strides = nullptr;
    } else {
        int64_t* strides = new int64_t[dl_tensor.ndim];
        for (int i = 0; i < dl_tensor.ndim; i++) {
            strides[i] = tensor_.stride(i);
        }
        dl_tensor.strides = strides;
    }
    // Set data type
    if (tensor_.scalar_type() == at::ScalarType::Float) {
        dl_tensor.dtype.code = DLDataTypeCode::kDLFloat;
        dl_tensor.dtype.bits = 32;
    } else if (tensor_.scalar_type() == at::ScalarType::Double) {
        dl_tensor.dtype.code = DLDataTypeCode::kDLFloat;
        dl_tensor.dtype.bits = 64;
    } else if (tensor_.scalar_type() == at::ScalarType::Int) {
        dl_tensor.dtype.code = DLDataTypeCode::kDLInt;
        dl_tensor.dtype.bits = 32;
    } else if (tensor_.scalar_type() == at::ScalarType::Long) {
        dl_tensor.dtype.code = DLDataTypeCode::kDLInt;
        dl_tensor.dtype.bits = 64;
    } else {
      throw std::runtime_error("Unsupported data type for DLPack conversion.");
    }
    dl_tensor.dtype.lanes = 1;
    // Point directly to the original tensor's data
    dl_tensor.data = tensor_.data_ptr();
    dl_tensor.byte_offset = 0;
    // Set up deleter
    managed->deleter = [](DLManagedTensorVersioned* self) {
        if (self) {
            // Free the shape array
            delete[] self->dl_tensor.shape;
            // Free the strides array if allocated
            if (self->dl_tensor.strides) {
                delete[] self->dl_tensor.strides;
            }
            delete static_cast<at::Tensor*>(self->manager_ctx);
            // Delete the managed tensor itself
            delete self;
        }
    };
    return managed;
}
