#ifndef METATENSOR_TORCH_MISC_HPP
#define METATENSOR_TORCH_MISC_HPP

#include <torch/script.h>

#include <metatensor.hpp>

#include "metatensor/torch/exports.h"
#include "metatensor/torch/tensor.hpp"

namespace metatensor_torch {

/// Get the runtime version of metatensor-torch as a string
METATENSOR_TORCH_EXPORT std::string version();

namespace details {
    /// Function to be used as `mts_create_array_callback_t` to load data in
    /// torch Tensor.
    METATENSOR_TORCH_EXPORT mts_status_t create_torch_array(
        const uintptr_t* shape_ptr,
        uintptr_t shape_count,
        mts_array_t* array
    );
}

/// Load a previously saved `TensorMap` from the given path.
METATENSOR_TORCH_EXPORT TorchTensorMap load(const std::string& path);

/// Load previously saved `Labels` from the given path.
METATENSOR_TORCH_EXPORT TorchLabels load_labels(const std::string& path);

/// Load a previously saved `TensorMap` from the given in-memory buffer
/// (represented as a `torch::Tensor` of bytes)
METATENSOR_TORCH_EXPORT TorchTensorMap load_buffer(torch::Tensor buffer);

/// Load previously saved `Labels` from the given in-memory buffer
/// (represented as a `torch::Tensor` of bytes)
METATENSOR_TORCH_EXPORT TorchLabels load_labels_buffer(torch::Tensor buffer);

/// Save the given `TensorMap` to a file at `path`
METATENSOR_TORCH_EXPORT void save(const std::string& path, TorchTensorMap tensor);

/// Save the given `Labels` to a file at `path`
METATENSOR_TORCH_EXPORT void save(const std::string& path, TorchLabels labels);

/// Save the given `TensorMap` to an in-memory buffer (represented as a
/// `torch::Tensor` of bytes)
METATENSOR_TORCH_EXPORT torch::Tensor save_buffer(TorchTensorMap tensor);

/// Save the given `Labels` to an in-memory buffer (represented as a
/// `torch::Tensor` of bytes)
METATENSOR_TORCH_EXPORT torch::Tensor save_buffer(TorchLabels labels);

}

#endif
