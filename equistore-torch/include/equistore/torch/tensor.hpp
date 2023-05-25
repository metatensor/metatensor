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
    TensorMapHolder(equistore::TensorMap tensor);

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
    TorchTensorBlock block_by_id(int64_t index);

    /// Get the block in this `TensorMap` with the key matching the name=>values
    /// passed in `selection`
    TorchTensorBlock block(const std::map<std::string, int32_t>& selection);

    TorchTensorBlock block(TorchLabels selection);
    TorchTensorBlock block(TorchLabelsEntry selection);

    /// TorchScript implementation of `block`, dispatching to one of the
    /// functions above
    TorchTensorBlock block_torch(torch::IValue index);

    /// Similar to `block_by_id`, but get all blocks with the given indices
    std::vector<TorchTensorBlock> blocks_by_id(const std::vector<int64_t>& indices);

    /// Get all blocks in this TensorMap
    std::vector<TorchTensorBlock> blocks();

    /// Similar to `block`, but allow getting multiple matching blocks
    std::vector<TorchTensorBlock> blocks(const std::map<std::string, int32_t>& selection);

    /// Similar to `block`, but allow getting multiple matching blocks
    std::vector<TorchTensorBlock> blocks(TorchLabels selection);
    std::vector<TorchTensorBlock> blocks(TorchLabelsEntry selection);

    /// TorchScript implementation of `blocks`, dispatching to one of the
    /// functions above.
    std::vector<TorchTensorBlock> blocks_torch(torch::IValue index);

    /// Merge blocks with the same value for selected keys dimensions along the
    /// property axis.
    ///
    /// See `equistore::TensorMap::keys_to_properties` for more information on
    /// this function.
    ///
    /// The input `torch::IValue` can be a single string, a list/tuple of
    /// strings, or a `TorchLabels` instance.
    TorchTensorMap keys_to_properties(torch::IValue keys_to_move, bool sort_samples) const;

    /// Merge blocks with the same value for selected keys dimensions along the
    /// sample axis.
    ///
    /// See `equistore::TensorMap::keys_to_samples` for more information on
    /// this function.
    ///
    /// The input `torch::IValue` can be a single string, a list/tuple of
    /// strings, or a `TorchLabels` instance.
    TorchTensorMap keys_to_samples(torch::IValue keys_to_move, bool sort_samples) const;

    /// Move the given `dimensions` from the component labels to the property
    /// labels for each block.
    ///
    /// See `equistore::TensorMap::components_to_properties` for more
    /// information on this function.
    ///
    /// The input `torch::IValue` can be a single string, or a list/tuple of
    /// strings.
    TorchTensorMap components_to_properties(torch::IValue dimensions) const;

    /// Get the sample names used for all block in this `TensorMap`
    std::vector<std::string> sample_names();

    /// Get the components names used for all block in this `TensorMap`
    std::vector<std::vector<std::string>> components_names();

    /// Get the property names used for all block in this `TensorMap`
    std::vector<std::string> property_names();

    /// Get all (key => block) pairs in this `TensorMap`
    std::vector<std::tuple<TorchLabelsEntry, TorchTensorBlock>> items();

    /// Print this TensorMap to a string, including at most `max_keys` in the
    /// output (-1 to include all keys).
    std::string print(int64_t max_keys) const;

    /// Get the underlying equistore TensorMap
    const equistore::TensorMap& as_equistore() const {
        return tensor_;
    }

private:
    /// Underlying equistore TensorMap
    equistore::TensorMap tensor_;
};


}

#endif
