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
using TorchTensorMap = torch::intrusive_ptr<TensorMapHolder>;

/// Wrapper around `metatensor::TensorMap` for integration with TorchScript
///
/// Python/TorchScript code will typically manipulate
/// `torch::intrusive_ptr<TensorMapHolder>` (i.e. `TorchTensorMap`) instead
/// of instances of `TensorMapHolder`.
class METATENSOR_TORCH_EXPORT TensorMapHolder: public torch::CustomClassHolder {
public:
    /// Wrap an existing `metatensor::TensorMap` into a `TensorMapHolder`
    explicit TensorMapHolder(metatensor::TensorMap tensor);

    /// Create a new `TensorMapHolder` for TorchScript.
    ///
    /// In contrast to the TensorMap constructor, this does not move from the
    /// different blocks, but instead create new ones using the same data and
    /// metadata, but with incremented reference count.
    TensorMapHolder(
        TorchLabels keys,
        const std::vector<TorchTensorBlock>& blocks
    );

    /// Make a copy of this `TensorMap`, including all the data contained inside
    TorchTensorMap copy() const;

    /// Get the keys for this `TensorMap`
    TorchLabels keys() const;

    /// Get a (possibly empty) list of block indexes matching the `selection`
    std::vector<int64_t> blocks_matching(const TorchLabels& selection) const;

    /// Get a block inside this TensorMap by it's index/the index of the
    /// corresponding key.
    ///
    /// The returned `TensorBlock` is a view inside memory owned by this
    /// `TensorMap`, and is only valid as long as the `TensorMap` is kept alive.
    static TorchTensorBlock block_by_id(TorchTensorMap self, int64_t index);

    /// Get the block in this `TensorMap` with the key matching the name=>values
    /// passed in `selection`
    static TorchTensorBlock block(TorchTensorMap self, const std::map<std::string, int32_t>& selection);

    /// Get the block in this `TensorMap` with the key matching the name=>values
    /// passed in `selection`. The `selection` must contain a single entry.
    static TorchTensorBlock block(TorchTensorMap self, TorchLabels selection);

    /// Get the block in this `TensorMap` with the key matching the name=>values
    /// passed in `selection`
    static TorchTensorBlock block(TorchTensorMap self, TorchLabelsEntry selection);

    /// TorchScript implementation of `block`, dispatching to one of the
    /// functions above
    static TorchTensorBlock block_torch(TorchTensorMap self, torch::IValue index);

    /// Similar to `block_by_id`, but get all blocks with the given indices
    static std::vector<TorchTensorBlock> blocks_by_id(TorchTensorMap self, const std::vector<int64_t>& indices);

    /// Get all blocks in this TensorMap
    static std::vector<TorchTensorBlock> blocks(TorchTensorMap self);

    /// Similar to `block`, but allow getting multiple matching blocks
    static std::vector<TorchTensorBlock> blocks(TorchTensorMap self, const std::map<std::string, int32_t>& selection);

    /// Similar to `block`, but allow getting multiple matching blocks
    static std::vector<TorchTensorBlock> blocks(TorchTensorMap self, TorchLabels selection);

    /// Similar to `block`, but allow getting multiple matching blocks
    static std::vector<TorchTensorBlock> blocks(TorchTensorMap self, TorchLabelsEntry selection);

    /// TorchScript implementation of `blocks`, dispatching to one of the
    /// functions above.
    static std::vector<TorchTensorBlock> blocks_torch(TorchTensorMap self, torch::IValue index);

    /// Merge blocks with the same value for selected keys dimensions along the
    /// property axis.
    ///
    /// See `metatensor::TensorMap::keys_to_properties` for more information on
    /// this function.
    ///
    /// The input `torch::IValue` can be a single string, a list/tuple of
    /// strings, or a `TorchLabels` instance.
    TorchTensorMap keys_to_properties(torch::IValue keys_to_move, bool sort_samples) const;

    /// Merge blocks with the same value for selected keys dimensions along the
    /// sample axis.
    ///
    /// See `metatensor::TensorMap::keys_to_samples` for more information on
    /// this function.
    ///
    /// The input `torch::IValue` can be a single string, a list/tuple of
    /// strings, or a `TorchLabels` instance.
    TorchTensorMap keys_to_samples(torch::IValue keys_to_move, bool sort_samples) const;

    /// Move the given `dimensions` from the component labels to the property
    /// labels for each block.
    ///
    /// See `metatensor::TensorMap::components_to_properties` for more
    /// information on this function.
    ///
    /// The input `torch::IValue` can be a single string, or a list/tuple of
    /// strings.
    TorchTensorMap components_to_properties(torch::IValue dimensions) const;

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
    static std::vector<std::tuple<TorchLabelsEntry, TorchTensorBlock>> items(TorchTensorMap self);

    /// Print this TensorMap to a string, including at most `max_keys` in the
    /// output (-1 to include all keys).
    std::string print(int64_t max_keys) const;

    /// Get the underlying metatensor TensorMap
    const metatensor::TensorMap& as_metatensor() const {
        return tensor_;
    }

private:
    /// Underlying metatensor TensorMap
    metatensor::TensorMap tensor_;
};


}

#endif
