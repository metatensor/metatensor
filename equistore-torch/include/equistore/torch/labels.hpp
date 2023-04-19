#ifndef EQUISTORE_TORCH_LABELS_HPP
#define EQUISTORE_TORCH_LABELS_HPP

#include <string>
#include <vector>

#include <torch/script.h>

#include <equistore.hpp>
#include "equistore/torch/exports.h"

namespace equistore_torch {

class LabelsHolder;
using TorchLabels = torch::intrusive_ptr<LabelsHolder>;

/// Wrapper around `equistore::Labels` for integration with TorchScript
///
/// Python/TorchScript code will typically manipulate
/// `torch::intrusive_ptr<LabelsHolder>` (i.e. `TorchLabels`) instead of
/// instances of `LabelsHolder`.
class EQUISTORE_TORCH_EXPORT LabelsHolder: public torch::CustomClassHolder {
public:
    LabelsHolder(std::vector<std::string> names, torch::Tensor values);

    // TODO

    equistore::Labels get() const {
        return labels_;
    }
private:
    equistore::Labels labels_;
};


}

#endif
