#ifndef METATENSOR_TORCH_MISC_HPP
#define METATENSOR_TORCH_MISC_HPP

#include <vector>

#include <torch/script.h>

#include <metatensor.hpp>

#include "metatensor/torch/exports.h"
#include "metatensor/torch/tensor.hpp"

namespace metatensor_torch {

/// Get the version of metatensor_torch
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

/// Save the given `TensorMap` to a file at `path`
METATENSOR_TORCH_EXPORT void save(const std::string& path, TorchTensorMap tensor);

}

#endif
