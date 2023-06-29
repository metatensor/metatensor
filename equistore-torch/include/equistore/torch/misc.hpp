#ifndef EQUISTORE_TORCH_MISC_HPP
#define EQUISTORE_TORCH_MISC_HPP

#include <equistore.h>
#include <vector>

#include <torch/script.h>

#include <equistore.hpp>

#include "equistore/torch/exports.h"
#include "equistore/torch/tensor.hpp"

namespace equistore_torch {

namespace details {
    /// Function to be used as `eqs_create_array_callback_t` to load data in
    /// torch Tensor.
    EQUISTORE_TORCH_EXPORT eqs_status_t create_torch_array(
        const uintptr_t* shape_ptr,
        uintptr_t shape_count,
        eqs_array_t* array
    );
}

/// Load a previously saved `TensorMap` from the given path.
EQUISTORE_TORCH_EXPORT TorchTensorMap load(const std::string& path);

/// Save the given `TensorMap` to a file at `path`
EQUISTORE_TORCH_EXPORT void save(const std::string& path, TorchTensorMap tensor);

}

#endif
