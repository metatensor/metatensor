#pragma once

#include <cassert>
#include <cstring>

#include <string>
#include <vector>
#include <optional>
#include <string_view>

#include <metatensor.h>

#include "./errors.hpp"
#include "./labels.hpp"
#include "./block.hpp"

namespace metatensor {

namespace details {

/// An iterator for TensorMap info key/values.
///
/// This class is not intended for direct usage, only through the
/// `TensorMap::info()` function. The iterator yields
/// `std::pair<std::string_view, std::string_view>`.
///
/// This class and any element obtained by iteration are only valid as long as
/// the underlying `TensorMap` is alive, and no changes are made to the info
/// map. If you need to keep the data longer, you should make a copy of the key
/// and value strings.
///
/// ```cpp
/// for (auto [key, value]: tensor_map.info()) {
///     // do something with key and value
/// }
/// ```
class TensorMapInfo {
public:
    /// @private Create an iterator for the given `tensor`
    explicit TensorMapInfo(mts_tensormap_t* tensor) : tensor_(tensor) {
        details::check_status(mts_tensormap_info_keys(
            tensor_,
            &keys_,
            &count_
        ));
    }

    /// @private
    class iterator {
    public:
        // iterator traits
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<std::string_view, std::string_view>;
        using pointer = const value_type*;
        using reference = value_type;
        using iterator_category = std::forward_iterator_tag;

        iterator& operator++() {
            index_ += 1;
            return *this;
        }

        iterator operator++(int) {
            iterator retval = *this;
            ++(*this);
            return retval;
        }

        bool operator==(iterator other) const {
            return tensor_ == other.tensor_ && index_ == other.index_;
        }

        bool operator!=(iterator other) const {return !(*this == other);}

        value_type operator*() {
            const char* value = nullptr;
            details::check_status(mts_tensormap_get_info(tensor_, keys_[index_], &value));
            return std::pair(keys_[index_], value);
        }

    private:
        friend class TensorMapInfo;
        iterator(mts_tensormap_t* tensor, const char* const* keys, uintptr_t index):
            tensor_(tensor), keys_(keys), index_(index) {}

        mts_tensormap_t* tensor_;
        const char* const* keys_;
        uintptr_t index_;
    };

    /// Get an iterator for the first element of the `TensorMap` info
    iterator begin() const {
        return iterator(tensor_, keys_, 0);
    }

    /// Get an iterator for the last element of the `TensorMap` info
    iterator end() const {
        return iterator(tensor_, nullptr, count_);
    }

private:
    mts_tensormap_t* tensor_;
    const char* const* keys_;
    uintptr_t count_;
};

}

/// A TensorMap is the main user-facing class of this library, and can store any
/// kind of data used in atomistic machine learning.
///
/// A tensor map contains a list of `TensorBlock`, each one associated with a
/// key. Users can access the blocks either one by one with the `block_by_id()`
/// function.
///
/// A tensor map provides functions to move some of these keys to the samples or
/// properties labels of the blocks, moving from a sparse representation of the
/// data to a dense one.
class TensorMap final {
public:
    /// Create a new TensorMap with the given `keys` and `blocks`
    TensorMap(Labels keys, std::vector<TensorBlock> blocks) {
        auto c_blocks = std::vector<mts_block_t*>();
        for (auto& block: blocks) {
            // We will move the data inside the new map, let's release the
            // pointers out of the TensorBlock now
            c_blocks.push_back(block.release());
        }

        tensor_ = mts_tensormap(
            keys.as_mts_labels_t(),
            c_blocks.data(),
            c_blocks.size()
        );

        details::check_pointer(tensor_);
    }

    ~TensorMap() {
        mts_tensormap_free(tensor_);
    }

    /// TensorMap can NOT be copy constructed, use TensorMap::clone instead
    TensorMap(const TensorMap&) = delete;
    /// TensorMap can not be copy assigned, use TensorMap::clone instead
    TensorMap& operator=(const TensorMap&) = delete;

    /// TensorMap can be move constructed
    TensorMap(TensorMap&& other) noexcept : TensorMap(nullptr) {
        *this = std::move(other);
    }

    /// TensorMap can be move assigned
    TensorMap& operator=(TensorMap&& other) noexcept {
        mts_tensormap_free(tensor_);

        this->tensor_ = other.tensor_;
        other.tensor_ = nullptr;

        return *this;
    }

    /// Make a copy of this `TensorMap`, including all the data contained inside
    TensorMap clone() const {
        auto* copy = mts_tensormap_copy(this->tensor_);
        details::check_pointer(copy);
        return TensorMap(copy);
    }

    /// Get a copy of the metadata in this `TensorMap` (i.e. keys, samples,
    /// components, and properties), ignoring the data itself.
    ///
    /// The resulting blocks values will be an `EmptyDataArray` instance, which
    /// does not contain any data.
    TensorMap clone_metadata_only() const {
        auto n_blocks = this->keys().count();

        auto blocks = std::vector<TensorBlock>();
        blocks.reserve(n_blocks);
        for (uintptr_t i=0; i<n_blocks; i++) {
            mts_block_t* block_ptr = nullptr;
            details::check_status(mts_tensormap_block_by_id(tensor_, &block_ptr, i));
            details::check_pointer(block_ptr);
            auto block = TensorBlock::unsafe_view_from_ptr(block_ptr);

            blocks.push_back(block.clone_metadata_only());
        }

        return TensorMap(this->keys(), std::move(blocks));
    }

    /// Get the set of keys labeling the blocks in this tensor map
    Labels keys() const {
        mts_labels_t keys;
        std::memset(&keys, 0, sizeof(keys));

        details::check_status(mts_tensormap_keys(tensor_, &keys));
        return Labels(keys);
    }

    /// Get a (possibly empty) list of block indexes matching the `selection`
    std::vector<uintptr_t> blocks_matching(const Labels& selection) const {
        auto matching = std::vector<uintptr_t>(this->keys().count());
        uintptr_t count = matching.size();

        details::check_status(mts_tensormap_blocks_matching(
            tensor_,
            matching.data(),
            &count,
            selection.as_mts_labels_t()
        ));

        assert(count <= matching.size());
        matching.resize(count);
        return matching;
    }

    /// Get a block inside this TensorMap by it's index/the index of the
    /// corresponding key.
    ///
    /// The returned `TensorBlock` is a view inside memory owned by this
    /// `TensorMap`, and is only valid as long as the `TensorMap` is kept alive.
    TensorBlock block_by_id(uintptr_t index) & {
        mts_block_t* block = nullptr;
        details::check_status(mts_tensormap_block_by_id(tensor_, &block, index));
        details::check_pointer(block);

        return TensorBlock::unsafe_view_from_ptr(block);
    }

    TensorBlock block_by_id(uintptr_t index) && = delete;

    /// Merge blocks with the same value for selected keys dimensions along the
    /// property axis.
    ///
    /// The dimensions (names) of `keys_to_move` will be moved from the keys to
    /// the property labels, and blocks with the same remaining keys dimensions
    /// will be merged together along the property axis.
    ///
    /// If `keys_to_move` does not contains any entries (i.e.
    /// `keys_to_move.count() == 0`), then the new property labels will contain
    /// entries corresponding to the merged blocks only. For example, merging a
    /// block with key `a=0` and properties `p=1, 2` with a block with key `a=2`
    /// and properties `p=1, 3` will produce a block with properties
    /// `a, p = (0, 1), (0, 2), (2, 1), (2, 3)`.
    ///
    /// If `keys_to_move` contains entries, then the property labels must be the
    /// same for all the merged blocks. In that case, the merged property labels
    /// will contains each of the entries of `keys_to_move` and then the current
    /// property labels. For example, using `a=2, 3` in `keys_to_move`, and
    /// blocks with properties `p=1, 2` will result in `a, p = (2, 1), (2, 2),
    /// (3, 1), (3, 2)`.
    ///
    /// The new sample labels will contains all of the merged blocks sample
    /// labels. The order of the samples is controlled by `sort_samples`. If
    /// `sort_samples` is true, samples are re-ordered to keep them
    /// lexicographically sorted. Otherwise they are kept in the order in which
    /// they appear in the blocks.
    ///
    /// @param keys_to_move description of the keys to move
    /// @param sort_samples whether to sort the merged samples or keep them in
    ///                     the order in which they appear in the original blocks
    TensorMap keys_to_properties(const Labels& keys_to_move, bool sort_samples = true) const {
        auto* ptr = mts_tensormap_keys_to_properties(
            tensor_,
            keys_to_move.as_mts_labels_t(),
            sort_samples
        );

        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    /// This function calls `keys_to_properties` with an empty set of `Labels`
    /// with the dimensions defined in `keys_to_move`
    TensorMap keys_to_properties(const std::vector<std::string>& keys_to_move, bool sort_samples = true) const {
        return keys_to_properties(Labels(keys_to_move), sort_samples);
    }

    /// This function calls `keys_to_properties` with an empty set of `Labels`
    /// with a single dimension: `key_to_move`
    TensorMap keys_to_properties(std::string key_to_move, bool sort_samples = true) const {
        return keys_to_properties(std::vector<std::string>{std::move(key_to_move)}, sort_samples);
    }

    /// Merge blocks with the same value for selected keys dimensions along the
    /// samples axis.
    ///
    /// The dimensions (names) of `keys_to_move` will be moved from the keys to
    /// the sample labels, and blocks with the same remaining keys dimensions
    /// will be merged together along the sample axis.
    ///
    /// If `keys_to_move` must be an empty set of `Labels`
    /// (`keys_to_move.count() == 0`). The new sample labels will contain
    /// entries corresponding to the merged blocks' keys.
    ///
    /// The order of the samples is controlled by `sort_samples`. If
    /// `sort_samples` is true, samples are re-ordered to keep them
    /// lexicographically sorted. Otherwise they are kept in the order in which
    /// they appear in the blocks.
    ///
    /// This function is only implemented if all merged block have the same
    /// property labels.
    ///
    /// @param keys_to_move description of the keys to move
    /// @param sort_samples whether to sort the merged samples or keep them in
    ///                     the order in which they appear in the original blocks
    TensorMap keys_to_samples(const Labels& keys_to_move, bool sort_samples = true) const {
        auto* ptr = mts_tensormap_keys_to_samples(
            tensor_,
            keys_to_move.as_mts_labels_t(),
            sort_samples
        );

        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    /// This function calls `keys_to_samples` with an empty set of `Labels`
    /// with the dimensions defined in `keys_to_move`
    TensorMap keys_to_samples(const std::vector<std::string>& keys_to_move, bool sort_samples = true) const {
        return keys_to_samples(Labels(keys_to_move), sort_samples);
    }

    /// This function calls `keys_to_samples` with an empty set of `Labels`
    /// with a single dimension: `key_to_move`
    TensorMap keys_to_samples(std::string key_to_move, bool sort_samples = true) const {
        return keys_to_samples(std::vector<std::string>{std::move(key_to_move)}, sort_samples);
    }

    /// Move the given `dimensions` from the component labels to the property
    /// labels for each block.
    ///
    /// @param dimensions name of the component dimensions to move to the
    ///                  properties
    TensorMap components_to_properties(const std::vector<std::string>& dimensions) const {
        auto c_dimensions = std::vector<const char*>();
        for (const auto& v: dimensions) {
            c_dimensions.push_back(v.c_str());
        }

        auto* ptr = mts_tensormap_components_to_properties(
            tensor_,
            c_dimensions.data(),
            c_dimensions.size()
        );
        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    /// Call `components_to_properties` with a single dimension
    TensorMap components_to_properties(const std::string& dimension) const {
        const char* c_str = dimension.c_str();
        auto* ptr = mts_tensormap_components_to_properties(
            tensor_,
            &c_str,
            1
        );
        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load a previously saved ``TensorMap`` from the given path.
     *
     * This is identical to :cpp:func:`metatensor::io::load`, and provided as a
     * convenience API.
     *
     * \endverbatim
     */
    static TensorMap load(
        const std::string& path,
        mts_create_array_callback_t create_array = details::default_create_array
    ) {
        return metatensor::io::load(path, create_array);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load a previously saved ``TensorMap`` from a in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::load_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    static TensorMap load_buffer(
        const uint8_t* buffer,
        size_t buffer_count,
        mts_create_array_callback_t create_array = details::default_create_array
    ) {
        return metatensor::io::load_buffer(buffer, buffer_count, create_array);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load a previously saved ``TensorMap`` from a in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::load_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    template <typename Buffer>
    static TensorMap load_buffer(
        const Buffer& buffer,
        mts_create_array_callback_t create_array = details::default_create_array
    ) {
        return metatensor::io::load_buffer<Buffer>(buffer, create_array);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Save this ``TensorMap`` to the given path.
     *
     * This is identical to :cpp:func:`metatensor::io::save`, and provided as a
     * convenience API.
     *
     * \endverbatim
     */
    void save(const std::string& path) const {
        metatensor::io::save(path, *this);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Save this ``TensorMap`` to an in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::save_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    std::vector<uint8_t> save_buffer() const {
        return metatensor::io::save_buffer(*this);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Save this ``TensorMap`` to an in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::save_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    template <typename Buffer>
    Buffer save_buffer() const {
        return metatensor::io::save_buffer<Buffer>(*this);
    }

    /// Get the `mts_tensormap_t` pointer corresponding to this `TensorMap`.
    ///
    /// The tensor map pointer is still managed by the current `TensorMap`
    mts_tensormap_t* as_mts_tensormap_t() & {
        return tensor_;
    }

    /// Get the const `mts_tensormap_t` pointer corresponding to this `TensorMap`.
    ///
    /// The tensor map pointer is still managed by the current `TensorMap`
    const mts_tensormap_t* as_mts_tensormap_t() const & {
        return tensor_;
    }

    mts_tensormap_t* as_mts_tensormap_t() && = delete;

    /// Create a C++ TensorMap from a C `mts_tensormap_t` pointer. The C++
    /// tensor map takes ownership of the C pointer.
    explicit TensorMap(mts_tensormap_t* tensor): tensor_(tensor) {}

    /// Set or update the info `value` associated with `key` for this `TensorMap`.
    void set_info(const std::string& key, const std::string& value) {
        details::check_status(mts_tensormap_set_info(tensor_, key.c_str(), value.c_str()));
    }

    /// Get the info value associated with `key` for this `TensorMap`.
    std::optional<std::string> get_info(const std::string& key) const {
        const char* value = nullptr;
        details::check_status(mts_tensormap_get_info(tensor_, key.c_str(), &value));

        if (value == nullptr) {
            return std::nullopt;
        } else {
            return std::string(value);
        }
    }

    /// Get an iterator over all the info key/values for this `TensorMap`.
    details::TensorMapInfo info() const {
        return details::TensorMapInfo(this->tensor_);
    }

private:
    mts_tensormap_t* tensor_;
};

} // namespace metatensor
