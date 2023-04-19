#ifndef EQUISTORE_TORCH_BLOCK_HPP
#define EQUISTORE_TORCH_BLOCK_HPP

#include <vector>

#include <torch/script.h>

#include <equistore.hpp>

#include "equistore/torch/exports.h"
#include "equistore/torch/labels.hpp"

namespace equistore_torch {

class TensorBlockHolder;
using TorchTensorBlock = torch::intrusive_ptr<TensorBlockHolder>;

/// Wrapper around `equistore::TensorBlock` for integration with TorchScript
///
/// Python/TorchScript code will typically manipulate
/// `torch::intrusive_ptr<TensorBlockHolder>` (i.e. `TorchTensorBlock`) instead
/// of instances of `TensorBlockHolder`.
class EQUISTORE_TORCH_EXPORT TensorBlockHolder: public torch::CustomClassHolder {
public:
    TensorBlockHolder(
        torch::Tensor data,
        TorchLabels samples,
        std::vector<TorchLabels> components,
        TorchLabels properties
    );

    // TODO
private:
    equistore::TensorBlock block_;
};

}

#endif
