#ifndef METATENSOR_TORCH_TENSOR_HPP
#define METATENSOR_TORCH_TENSOR_HPP

#include <vector>

#include <torch/script.h>

#include <metatensor.hpp>

#include "metatensor/torch/exports.h"
#include "metatensor/torch/labels.hpp"
#include "metatensor/torch/block.hpp"

namespace metatensor_torch {

class TensorMapHolder;
/// TorchScript will always manipulate `TensorMapHolder` through a `torch::intrusive_ptr`
using TensorMap = torch::intrusive_ptr<TensorMapHolder>;

// for backward compatibility, to remove later
using TorchTensorMap = TensorMap;

/// Wrapper around `metatensor::TensorMap` for integration with TorchScript
///
/// Python/TorchScript code will typically manipulate
/// `torch::intrusive_ptr<TensorMapHolder>` (i.e. `TensorMap`) instead
/// of instances of `TensorMapHolder`.
class METATENSOR_TORCH_EXPORT TensorMapHolder: public torch::CustomClassHolder {
public:
    /// Create a new `TensorMapHolder` for TorchScript.
    ///
    /// In contrast to the TensorMap constructor, this does not move from the
    /// different blocks, but instead create new ones using the same data and
    /// metadata, but with incremented reference count.
    TensorMapHolder(
        Labels keys,
        const std::vector<TensorBlock>& blocks
    );

    /// Make a copy of this `TensorMap`, including all the data contained inside
    TensorMap copy() const;

    /// Get the keys for this `TensorMap`
    Labels keys() const;

    /// Get a (possibly empty) list of block indexes matching the `selection`
    std::vector<int64_t> blocks_matching(const Labels& selection) const;

    /// Get a block inside this TensorMap by it's index/the index of the
    /// corresponding key.
    ///
    /// The returned `TensorBlock` is a view inside memory owned by this
    /// `TensorMap`, and is only valid as long as the `TensorMap` is kept alive.
    static TensorBlock block_by_id(TensorMap self, int64_t index);

    /// Get the block in this `TensorMap` with the key matching the name=>values
    /// passed in `selection`
    static TensorBlock block(TensorMap self, const std::map<std::string, int32_t>& selection);

    /// Get the block in this `TensorMap` with the key matching the name=>values
    /// passed in `selection`. The `selection` must contain a single entry.
    static TensorBlock block(TensorMap self, Labels selection);

    /// Get the block in this `TensorMap` with the key matching the name=>values
    /// passed in `selection`
    static TensorBlock block(TensorMap self, LabelsEntry selection);

    /// TorchScript implementation of `block`, dispatching to one of the
    /// functions above
    static TensorBlock block_torch(TensorMap self, torch::IValue index);

    /// Similar to `block_by_id`, but get all blocks with the given indices
    static std::vector<TensorBlock> blocks_by_id(TensorMap self, const std::vector<int64_t>& indices);

    /// Get all blocks in this TensorMap
    static std::vector<TensorBlock> blocks(TensorMap self);

    /// Similar to `block`, but allow getting multiple matching blocks
    static std::vector<TensorBlock> blocks(TensorMap self, const std::map<std::string, int32_t>& selection);

    /// Similar to `block`, but allow getting multiple matching blocks
    static std::vector<TensorBlock> blocks(TensorMap self, Labels selection);

    /// Similar to `block`, but allow getting multiple matching blocks
    static std::vector<TensorBlock> blocks(TensorMap self, LabelsEntry selection);

    /// TorchScript implementation of `blocks`, dispatching to one of the
    /// functions above.
    static std::vector<TensorBlock> blocks_torch(TensorMap self, torch::IValue index);

    /// Merge blocks with the same value for selected keys dimensions along the
    /// property axis.
    ///
    /// See `metatensor::TensorMap::keys_to_properties` for more information on
    /// this function.
    ///
    /// The input `torch::IValue` can be a single string, a list/tuple of
    /// strings, or a `Labels` instance.
    TensorMap keys_to_properties(torch::IValue keys_to_move, bool sort_samples) const;

    /// Merge blocks with the same value for selected keys dimensions along the
    /// sample axis.
    ///
    /// See `metatensor::TensorMap::keys_to_samples` for more information on
    /// this function.
    ///
    /// The input `torch::IValue` can be a single string, a list/tuple of
    /// strings, or a `Labels` instance.
    TensorMap keys_to_samples(torch::IValue keys_to_move, bool sort_samples) const;

    /// Move the given `dimensions` from the component labels to the property
    /// labels for each block.
    ///
    /// See `metatensor::TensorMap::components_to_properties` for more
    /// information on this function.
    ///
    /// The input `torch::IValue` can be a single string, or a list/tuple of
    /// strings.
    TensorMap components_to_properties(torch::IValue dimensions) const;

    /// Get the names of the samples dimensions for all blocks in this
    /// `TensorMap`
    std::vector<std::string> sample_names();

    /// Get the names of the components dimensions for all blocks in this
    /// `TensorMap`
    std::vector<std::string> component_names();

    /// Get the names of the properties dimensions for all blocks in this
    /// `TensorMap`
    std::vector<std::string> property_names();

    /// Get all (key => block) pairs in this `TensorMap`
    static std::vector<std::tuple<LabelsEntry, TensorBlock>> items(TensorMap self);

    /// Print this TensorMap to a string, including at most `max_keys` in the
    /// output (-1 to include all keys).
    std::string print(int64_t max_keys) const;

    /// Get the device for the values stored in this `TensorMap`
    torch::Device device() const;

    /// Get the dtype for the values stored in this `TensorMap`
    torch::Dtype scalar_type() const;

    /// Move this `TensorMap` to the given `dtype` and `device`.
    TensorMap to(
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
    TensorMap to_positional(
        torch::IValue positional_1,
        torch::IValue positional_2,
        torch::optional<torch::Dtype> dtype,
        torch::optional<torch::Device> device,
        torch::optional<std::string> arrays
    ) const;

    /// Get the underlying metatensor TensorMap
    const metatensor::TensorMap& as_metatensor() const {
        return tensor_;
    }

    /// Load a serialized TensorMap from the given path
    static TensorMap load(const std::string& path);

    /// Load a serialized TensorMap from an in-memory buffer (represented as a
    /// `torch::Tensor` of bytes)
    static TensorMap load_buffer(torch::Tensor buffer);

    /// Serialize and save a TensorMap to the given path
    void save(const std::string& path) const;

    /// Serialize and save a TensorMap to an in-memory buffer (represented as a
    /// `torch::Tensor` of bytes)
    torch::Tensor save_buffer() const;

private:
    /// Underlying metatensor TensorMap
    metatensor::TensorMap tensor_;

    /// Wrap an existing `metatensor::TensorMap` into a `TensorMapHolder`
    explicit TensorMapHolder(metatensor::TensorMap tensor): tensor_(std::move(tensor)) {}
};


}

#endif
