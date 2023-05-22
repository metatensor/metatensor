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
    /// Create a new TensorBlockHolder with the given data and metadata
    TensorBlockHolder(
        torch::Tensor data,
        TorchLabels samples,
        std::vector<TorchLabels> components,
        TorchLabels properties
    );

    /// Create a torch TensorBlockHolder from a pre-existing equistore::TensorBlock
    TensorBlockHolder(equistore::TensorBlock block);

    /// Make a copy of this `TensorBlockHolder`, including all the data
    /// contained inside
    TorchTensorBlock copy() const;

    /// Get a view in the values in this block
    torch::Tensor values();

    /// Get the labels in this block associated with either `"values"` or one
    /// gradient (by setting `values_gradients` to the gradient parameter); in
    /// the given `axis`.
    TorchLabels labels(uintptr_t axis) const;

    /// Access the sample `Labels` for this block.
    ///
    /// The entries in these labels describe the first dimension of the
    /// `values()` array.
    TorchLabels samples() const {
        return this->labels(0);
    }

    /// Access the component `Labels` for this block.
    ///
    /// The entries in these labels describe intermediate dimensions of the
    /// `values()` array.
    std::vector<TorchLabels> components() const {
        auto shape = this->block_.values_shape();

        auto result = std::vector<TorchLabels>();
        for (size_t i=1; i<shape.size() - 1; i++) {
            result.emplace_back(this->labels(i));
        }

        return result;
    }

    /// Access the property `Labels` for this block.
    ///
    /// The entries in these labels describe the last dimension of the
    /// `values()` array. The properties are guaranteed to be the same for
    /// values and gradients in the same block.
    TorchLabels properties() const {
        auto shape = this->block_.values_shape();
        return this->labels(shape.size() - 1);
    }

    /// Add a set of gradients with respect to `parameters` in this block.
    ///
    /// @param parameter add gradients with respect to this `parameter` (e.g.
    ///                 `"positions"`, `"cell"`, ...)
    /// @param gradient a `TorchTensorBlock` whose values contain the gradients
    ///                 with respect to the `parameter`. The labels of the
    ///                 gradient `TorchTensorBlock` should be organized as
    ///                 follows: its `samples` must contain `"sample"` as the
    ///                 first label, which establishes a correspondence with the
    ///                 `samples` of the original `TorchTensorBlock`; its
    ///                 components must contain at least the same components as
    ///                 the original `TorchTensorBlock`, with any additional
    ///                 component coming before those; its properties must match
    ///                 those of the original `TorchTensorBlock`.
    void add_gradient(const std::string& parameter, TorchTensorBlock gradient);

    /// Get a list of all gradients defined in this block.
    std::vector<std::string> gradients_list() const {
        return block_.gradients_list();
    }

    /// Check if a given gradient is defined in this TensorBlock
    bool has_gradient(const std::string& parameter) const;

    // Get a gradient from this TensorBlock
    TorchTensorBlock gradient(const std::string& parameter) const;

    /// Get a all gradients and associated parameters in this block
    std::unordered_map<std::string, TorchTensorBlock> gradients();

    /// Implementation of __repr__/__str__ for Python
    std::string __repr__() const;

    /// Get the underlying equistore TensorBlock
    const equistore::TensorBlock& as_equistore() const {
        return block_;
    }

private:
    /// Create a TensorBlockHolder containing gradients with respect to
    /// `parameter`
    TensorBlockHolder(equistore::TensorBlock block, std::string parameter);
    friend class torch::intrusive_ptr<TensorBlockHolder>;

    /// Underlying equistore TensorBlock
    equistore::TensorBlock block_;
    /// If this TensorBlock contains gradients, these are gradients w.r.t. this
    /// parameter
    std::string parameter_;
};

}

#endif
