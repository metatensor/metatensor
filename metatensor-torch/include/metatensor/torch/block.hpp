#ifndef METATENSOR_TORCH_BLOCK_HPP
#define METATENSOR_TORCH_BLOCK_HPP

#include <vector>

#include <torch/script.h>

#include <metatensor.hpp>

#include "metatensor/torch/exports.h"
#include "metatensor/torch/labels.hpp"

namespace metatensor_torch {

class TensorBlockHolder;
/// TorchScript will always manipulate `TensorBlockHolder` through a `torch::intrusive_ptr`
using TensorBlock = torch::intrusive_ptr<TensorBlockHolder>;

// for backward compatibility, to remove later
using TorchTensorBlock = TensorBlock;

/// Wrapper around `metatensor::TensorBlock` for integration with TorchScript
///
/// Python/TorchScript code will typically manipulate
/// `torch::intrusive_ptr<TensorBlockHolder>` (i.e. `TensorBlock`) instead
/// of instances of `TensorBlockHolder`.
class METATENSOR_TORCH_EXPORT TensorBlockHolder: public torch::CustomClassHolder {
public:
    /// Create a new TensorBlockHolder with the given data and metadata
    TensorBlockHolder(
        torch::Tensor data,
        Labels samples,
        std::vector<Labels> components,
        Labels properties
    );

    /// Create a torch TensorBlockHolder from a pre-existing
    /// `metatensor::TensorBlock`.
    ///
    /// If the block is a view inside another `TensorBlock` or
    /// `TensorMap`, then `parent` should point to the corresponding
    /// object, making sure a reference to it is kept around.
    TensorBlockHolder(metatensor::TensorBlock block, torch::IValue parent);

    /// Make a copy of this `TensorBlockHolder`, including all the data
    /// contained inside
    TensorBlock copy() const;

    /// Get a view in the values in this block
    torch::Tensor values() const;

    /// Get the labels in this block associated with either `"values"` or one
    /// gradient (by setting `values_gradients` to the gradient parameter); in
    /// the given `axis`.
    Labels labels(uintptr_t axis) const;

    /// Access the sample `Labels` for this block.
    ///
    /// The entries in these labels describe the first dimension of the
    /// `values()` array.
    Labels samples() const {
        return this->labels(0);
    }

    /// Get the length of this block, i.e. the number of samples
    int64_t len() const {
        return this->labels(0)->count();
    }

    /// Get the shape of the values Tensor
    at::IntArrayRef shape() const{
        return this->values().sizes() ;
    }

    /// Access the component `Labels` for this block.
    ///
    /// The entries in these labels describe intermediate dimensions of the
    /// `values()` array.
    std::vector<Labels> components() const {
        auto shape = this->block_.values_shape();

        auto result = std::vector<Labels>();
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
    Labels properties() const {
        auto shape = this->block_.values_shape();
        return this->labels(shape.size() - 1);
    }

    /// Add a set of gradients with respect to `parameters` in this block.
    ///
    /// @param parameter add gradients with respect to this `parameter` (e.g.
    ///                 `"positions"`, `"cell"`, ...)
    /// @param gradient a `TensorBlock` whose values contain the gradients
    ///                 with respect to the `parameter`. The labels of the
    ///                 gradient `TensorBlock` should be organized as
    ///                 follows: its `samples` must contain `"sample"` as the
    ///                 first label, which establishes a correspondence with the
    ///                 `samples` of the original `TensorBlock`; its
    ///                 components must contain at least the same components as
    ///                 the original `TensorBlock`, with any additional
    ///                 component coming before those; its properties must match
    ///                 those of the original `TensorBlock`.
    void add_gradient(const std::string& parameter, TensorBlock gradient);

    /// Get a list of all gradients defined in this block.
    std::vector<std::string> gradients_list() const {
        return block_.gradients_list();
    }

    /// Check if a given gradient is defined in this TensorBlock
    bool has_gradient(const std::string& parameter) const;

    /// Get a gradient from this TensorBlock
    static TensorBlock gradient(TensorBlock self, const std::string& parameter);

    /// Get a all gradients and associated parameters in this block
    static std::vector<std::tuple<std::string, TensorBlock>> gradients(TensorBlock self);

    /// Get the device for the values stored in this `TensorBlock`
    torch::Device device() const {
        return this->values().device();
    }

    /// Get the dtype for the values stored in this `TensorBlock`
    torch::Dtype scalar_type() const {
        return this->values().scalar_type();
    }

    /// Move all arrays in this block to the given `dtype` and `device`.
    TensorBlock to(
        torch::optional<torch::Dtype> dtype = torch::nullopt,
        torch::optional<torch::Device> device = torch::nullopt
    ) const;

    /// Wrapper of the `to` function to enable using it with positional
    /// parameters from Python; for example `to(dtype)`, `to(device)`,
    /// `to(dtype, device=device)`, `to(dtype, device)`, `to(device, dtype)`,
    /// etc.
    ///
    /// `arrays` is left as a keyword argument since it is mainly here for
    /// compatibility with the pure Python backend, and only `"torch"` is
    /// supported.
    TensorBlock to_positional(
        torch::IValue positional_1,
        torch::IValue positional_2,
        torch::optional<torch::Dtype> dtype,
        torch::optional<torch::Device> device,
        torch::optional<std::string> arrays
    ) const;

    /// Implementation of __repr__/__str__ for Python
    std::string repr() const;

    /// Get the underlying metatensor TensorBlock
    const metatensor::TensorBlock& as_metatensor() const {
        return block_;
    }

    /// Load a serialized TensorBlock from the given path
    static TensorBlock load(const std::string& path);

    /// Load a serialized TensorBlock from an in-memory buffer (represented as a
    /// `torch::Tensor` of bytes)
    static TensorBlock load_buffer(torch::Tensor buffer);

    /// Serialize and save a TensorBlock to the given path
    void save(const std::string& path) const;

    /// Serialize and save a TensorBlock to an in-memory buffer (represented as
    /// a `torch::Tensor` of bytes)
    torch::Tensor save_buffer() const;

    // Setter that throw and error
    void set_values(const torch::Tensor& new_values);

private:
    /// Create a TensorBlockHolder containing gradients with respect to
    /// `parameter`
    TensorBlockHolder(metatensor::TensorBlock block, std::string parameter, torch::IValue parent);
    friend class torch::intrusive_ptr<TensorBlockHolder>;

    /// Underlying metatensor TensorBlock
    metatensor::TensorBlock block_;

    /// Parent for this block, either `None`, another `TensorBlock` (if this
    /// block contains gradients), or a `TensorMap`.
    torch::IValue parent_;

    /// If this TensorBlock contains gradients, these are gradients w.r.t. this
    /// parameter
    std::string parameter_;
};

}

#endif
