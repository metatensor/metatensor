#ifndef EQUISTORE_TORCH_TENSOR_HPP
#define EQUISTORE_TORCH_TENSOR_HPP

#include <vector>

#include <torch/script.h>

#include <equistore.hpp>

#include "equistore/torch/exports.h"
#include "equistore/torch/labels.hpp"
#include "equistore/torch/block.hpp"

namespace equistore_torch {

class TensorMapHolder;
using TorchTensorMap = torch::intrusive_ptr<TensorMapHolder>;

/// Wrapper around `equistore::TensorMap` for integration with TorchScript
///
/// Python/TorchScript code will typically manipulate
/// `torch::intrusive_ptr<TensorMapHolder>` (i.e. `TorchTensorMap`) instead
/// of instances of `TensorMapHolder`.
class EQUISTORE_TORCH_EXPORT TensorMapHolder: public torch::CustomClassHolder {
public:
    TensorMapHolder(
        TorchLabels keys,
        std::vector<TorchTensorBlock> blocks
    );

    /// TODO
private:
    equistore::TensorMap tensor_;
};

}

#endif
