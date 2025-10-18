#pragma once

#include <array>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include <metatensor.h>

#include "./errors.hpp"
#include "./arrays.hpp"
#include "./io_fwd.hpp"

namespace metatensor_torch {
    class LabelsHolder;
}

namespace metatensor {
    class Labels;

    namespace details {
        Labels labels_from_cxx(
            const std::vector<std::string>& names,
            const int32_t* values,
            size_t count,
            bool assume_unique
        );
    }

/// It is possible to store some user-provided data inside `Labels`, and access
/// it later. This class is used to take ownership of the data and corresponding
/// delete function before giving the data to metatensor.
///
/// User data inside `Labels` is an advanced functionality, that most users
/// should not need to interact with.
class LabelsUserData;

/// Tag for the creation of Labels without uniqueness checks
struct assume_unique {};

/// A set of labels used to carry metadata associated with a tensor map.
///
/// This is similar to an array of named tuples, but stored as a 2D array
/// of shape `(count, size)`, with a set of names associated with the columns of
/// this array (often called *dimensions*). Each row/entry in this array is
/// unique, and they are often (but not always) sorted in lexicographic order.
class Labels final {
public:
    /// Create a new set of Labels from the given `names` and `values`.
    ///
    /// Each entry in the values must contain `names.size()` elements.
    ///
    /// ```
    /// auto labels = Labels({"first", "second"}, {
    ///    {0, 1},
    ///    {1, 4},
    ///    {2, 1},
    ///    {2, 3},
    /// });
    /// ```
    Labels(
        const std::vector<std::string>& names,
        const std::vector<std::initializer_list<int32_t>>& values
    ): Labels(names, NDArray<int32_t>(values, names.size()), InternalConstructor{}) {}

    /// This function does not check for uniqueness of the labels entries, which
    /// should be enforced by the caller. Calling this function with non-unique
    /// entries is invalid and can lead to crashes or infinite loops.
    explicit Labels(
        const std::vector<std::string>& names,
        const std::vector<std::initializer_list<int32_t>>& values,
        assume_unique
    ): Labels(names, NDArray<int32_t>(values, names.size()), assume_unique{}, InternalConstructor{}) {}

    /// Create an empty set of Labels with the given names
    explicit Labels(const std::vector<std::string>& names):
        Labels(names, static_cast<const int32_t*>(nullptr), 0) {}

    /// Create labels with the given `names` and `values`. `values` must be an
    /// array with `count x names.size()` elements.
    Labels(const std::vector<std::string>& names, const int32_t* values, size_t count):
        Labels(details::labels_from_cxx(names, values, count, false)) {}

    /// Unchecked variant, caller promises the labels are unique. Calling with
    /// non-unique entries is invalid and can ead to crashes or infinite loops.
    Labels(const std::vector<std::string>& names, const int32_t* values, size_t count, assume_unique):
        Labels(details::labels_from_cxx(names, values, count, true)) {}

    ~Labels() {
        mts_labels_free(&labels_);
    }

    /// Labels is copy-constructible
    Labels(const Labels& other): Labels() {
        *this = other;
    }

    /// Labels can be copy-assigned
    Labels& operator=(const Labels& other) {
        mts_labels_free(&labels_);
        std::memset(&labels_, 0, sizeof(labels_));
        details::check_status(mts_labels_clone(other.labels_, &labels_));
        assert(this->labels_.internal_ptr_ != nullptr);

        this->values_ = NDArray<int32_t>(labels_.values, {labels_.count, labels_.size});

        this->names_.clear();
        for (size_t i=0; i<this->labels_.size; i++) {
            this->names_.push_back(this->labels_.names[i]);
        }

        return *this;
    }

    /// Labels is move-constructible
    Labels(Labels&& other) noexcept: Labels() {
        *this = std::move(other);
    }

    /// Labels can be move-assigned
    Labels& operator=(Labels&& other) noexcept {
        mts_labels_free(&labels_);
        this->labels_ = other.labels_;
        assert(this->labels_.internal_ptr_ != nullptr);
        std::memset(&other.labels_, 0, sizeof(other.labels_));

        this->values_ = std::move(other.values_);
        this->names_ = std::move(other.names_);

        return *this;
    }

    /// Get the names of the dimensions used in these `Labels`.
    const std::vector<const char*>& names() const {
        return names_;
    }

    /// Get the number of entries in this set of Labels.
    ///
    /// This is the same as `shape()[0]` for the corresponding values array
    size_t count() const {
        return labels_.count;
    }

    /// Get the number of dimensions in this set of Labels.
    ///
    /// This is the same as `shape()[1]` for the corresponding values array
    size_t size() const {
        return labels_.size;
    }

    /// Convert from this set of Labels to the C `mts_labels_t`
    mts_labels_t as_mts_labels_t() const {
        assert(labels_.internal_ptr_ != nullptr);
        return labels_;
    }

    /// Get the user data pointer registered with these `Labels`.
    ///
    /// If no user data have been registered, this function will return
    /// `nullptr`.
    void* user_data() & {
        assert(labels_.internal_ptr_ != nullptr);

        void* data = nullptr;
        details::check_status(mts_labels_user_data(labels_, &data));
        return data;
    }

    void* user_data() && = delete;

    /// Register some user data pointer with these `Labels`.
    ///
    /// Any existing user data will be released (by calling the provided
    /// `delete` function) before overwriting with the new data.
    void set_user_data(LabelsUserData user_data);

    /// Get the position of the `entry` in this set of Labels, or -1 if the
    /// entry is not part of these Labels.
    int64_t position(std::initializer_list<int32_t> entry) const {
        return this->position(entry.begin(), entry.size());
    }

    /// Variant of `Labels::position` taking a fixed-size array as input
    template<size_t N>
    int64_t position(const std::array<int32_t, N>& entry) const {
        return this->position(entry.data(), entry.size());
    }

    /// Variant of `Labels::position` taking a vector as input
    int64_t position(const std::vector<int32_t>& entry) const {
        return this->position(entry.data(), entry.size());
    }

    /// Variant of `Labels::position` taking a pointer and length as input
    int64_t position(const int32_t* entry, size_t length) const {
        assert(labels_.internal_ptr_ != nullptr);

        int64_t result = 0;
        details::check_status(mts_labels_position(labels_, entry, length, &result));
        return result;
    }

    /// Get the array of values for these Labels
    const NDArray<int32_t>& values() const & {
        return values_;
    }

    const NDArray<int32_t>& values() && = delete;

    /// Take the union of these `Labels` with `other`.
    ///
    /// If requested, this function can also give the positions in the union
    /// where each entry of the input `Labels` ended up.
    ///
    /// No user data pointer is registered with the output, even if the inputs
    /// have some.
    ///
    /// @param other the `Labels` we want to take the union with
    /// @param first_mapping if you want the mapping from the positions of
    ///        entries in `this` to the positions in the union, this should be
    ///        a pointer to an array containing `this->count()` elements, to be
    ///        filled by this function. Otherwise it should be a `nullptr`.
    /// @param first_mapping_count number of elements in `first_mapping`
    /// @param second_mapping if you want the mapping from the positions of
    ///        entries in `other` to the positions in the union, this should be
    ///        a pointer to an array containing `other.count()` elements, to be
    ///        filled by this function. Otherwise it should be a `nullptr`.
    /// @param second_mapping_count number of elements in `second_mapping`
    Labels set_union(
        const Labels& other,
        int64_t* first_mapping = nullptr,
        size_t first_mapping_count = 0,
        int64_t* second_mapping = nullptr,
        size_t second_mapping_count = 0
    ) const {
        mts_labels_t result;
        std::memset(&result, 0, sizeof(result));

        details::check_status(mts_labels_union(
            labels_,
            other.labels_,
            &result,
            first_mapping,
            first_mapping_count,
            second_mapping,
            second_mapping_count
        ));

        return Labels(result);
    }

    /// Take the union of these `Labels` with `other`.
    ///
    /// If requested, this function can also give the positions in the
    /// union where each entry of the input `Labels` ended up.
    ///
    /// No user data pointer is registered with the output, even if the inputs
    /// have some.
    ///
    /// @param other the `Labels` we want to take the union with
    /// @param first_mapping if you want the mapping from the positions of
    ///        entries in `this` to the positions in the union, this should be
    ///        a vector containing `this->count()` elements, to be filled by
    ///        this function. Otherwise it should be an empty vector.
    /// @param second_mapping if you want the mapping from the positions of
    ///        entries in `other` to the positions in the union, this should be
    ///        a vector containing `other.count()` elements, to be filled by
    ///        this function. Otherwise it should be an empty vector.
    Labels set_union(
        const Labels& other,
        std::vector<int64_t>& first_mapping,
        std::vector<int64_t>& second_mapping
    ) const {
        auto* first_mapping_ptr = first_mapping.data();
        auto first_mapping_count = first_mapping.size();
        if (first_mapping_count == 0) {
            first_mapping_ptr = nullptr;
        }

        auto* second_mapping_ptr = second_mapping.data();
        auto second_mapping_count = second_mapping.size();
        if (second_mapping_count == 0) {
            second_mapping_ptr = nullptr;
        }

        return this->set_union(
            other,
            first_mapping_ptr,
            first_mapping_count,
            second_mapping_ptr,
            second_mapping_count
        );
    }

    /// Take the intersection of these `Labels` with `other`.
    ///
    /// If requested, this function can also give the positions in the
    /// intersection where each entry of the input `Labels` ended up.
    ///
    /// No user data pointer is registered with the output, even if the inputs
    /// have some.
    ///
    /// @param other the `Labels` we want to take the intersection with
    /// @param first_mapping if you want the mapping from the positions of
    ///        entries in `this` to the positions in the intersection, this
    ///        should be a pointer to an array containing `this->count()`
    ///        elements, to be filled by this function. Otherwise it should be a
    ///        `nullptr`. If an entry in `this` is not used in the intersection,
    ///        the mapping will be set to -1.
    /// @param first_mapping_count number of elements in `first_mapping`
    /// @param second_mapping if you want the mapping from the positions of
    ///        entries in `other` to the positions in the intersection, this
    ///        should be a pointer to an array containing `other.count()`
    ///        elements, to be filled by this function. Otherwise it should be a
    ///        `nullptr`. If an entry in `other` is not used in the
    ///        intersection, the mapping will be set to -1.
    /// @param second_mapping_count number of elements in `second_mapping`
    Labels set_intersection(
        const Labels& other,
        int64_t* first_mapping = nullptr,
        size_t first_mapping_count = 0,
        int64_t* second_mapping = nullptr,
        size_t second_mapping_count = 0
    ) const {
        mts_labels_t result;
        std::memset(&result, 0, sizeof(result));

        details::check_status(mts_labels_intersection(
            labels_,
            other.labels_,
            &result,
            first_mapping,
            first_mapping_count,
            second_mapping,
            second_mapping_count
        ));

        return Labels(result);
    }

    /// Take the intersection of this `Labels` with `other`.
    ///
    /// If requested, this function can also give the positions in the
    /// intersection where each entry of the input `Labels` ended up.
    ///
    /// No user data pointer is registered with the output, even if the inputs
    /// have some.
    ///
    /// @param other the `Labels` we want to take the intersection with
    /// @param first_mapping if you want the mapping from the positions of
    ///        entries in `this` to the positions in the intersection, this
    ///        should be a vector containing `this->count()` elements, to be
    ///        filled by this function. Otherwise it should be an empty vector.
    ///        If an entry in `this` is not used in the intersection, the
    ///        mapping will be set to -1.
    /// @param second_mapping if you want the mapping from the positions of
    ///        entries in `other` to the positions in the intersection, this
    ///        should be a vector containing `other.count()` elements, to be
    ///        filled by this function. Otherwise it should be an empty vector.
    ///        If an entry in `other` is not used in the intersection, the
    ///        mapping will be set to -1.
    Labels set_intersection(
        const Labels& other,
        std::vector<int64_t>& first_mapping,
        std::vector<int64_t>& second_mapping
    ) const {
        auto* first_mapping_ptr = first_mapping.data();
        auto first_mapping_count = first_mapping.size();
        if (first_mapping_count == 0) {
            first_mapping_ptr = nullptr;
        }

        auto* second_mapping_ptr = second_mapping.data();
        auto second_mapping_count = second_mapping.size();
        if (second_mapping_count == 0) {
            second_mapping_ptr = nullptr;
        }

        return this->set_intersection(
            other,
            first_mapping_ptr,
            first_mapping_count,
            second_mapping_ptr,
            second_mapping_count
        );
    }

    /// Take the difference of these `Labels` with `other`.
    ///
    /// If requested, this function can also give the positions in the
    /// difference where each entry of the input `Labels` ended up.
    ///
    /// No user data pointer is registered with the output, even if the inputs
    /// have some.
    ///
    /// @param other the `Labels` we want to take the difference with
    /// @param first_mapping if you want the mapping from the positions of
    ///        entries in `this` to the positions in the difference, this
    ///        should be a pointer to an array containing `this->count()`
    ///        elements, to be filled by this function. Otherwise it should be a
    ///        `nullptr`. If an entry in `this` is not used in the difference,
    ///        the mapping will be set to -1.
    /// @param first_mapping_count number of elements in `first_mapping`
    Labels set_difference(
        const Labels& other,
        int64_t* first_mapping = nullptr,
        size_t first_mapping_count = 0
    ) const {
        mts_labels_t result;
        std::memset(&result, 0, sizeof(result));

        details::check_status(mts_labels_difference(
            labels_,
            other.labels_,
            &result,
            first_mapping,
            first_mapping_count
        ));

        return Labels(result);
    }

    /// Take the difference of this `Labels` with `other`.
    ///
    /// If requested, this function can also give the positions in the
    /// difference where each entry of the input `Labels` ended up.
    ///
    /// No user data pointer is registered with the output, even if the inputs
    /// have some.
    ///
    /// @param other the `Labels` we want to take the difference with
    /// @param first_mapping if you want the mapping from the positions of
    ///        entries in `this` to the positions in the difference, this
    ///        should be a vector containing `this->count()` elements, to be
    ///        filled by this function. Otherwise it should be an empty vector.
    ///        If an entry in `this` is not used in the difference, the
    ///        mapping will be set to -1.
    Labels set_difference(const Labels& other, std::vector<int64_t>& first_mapping) const {
        auto* first_mapping_ptr = first_mapping.data();
        auto first_mapping_count = first_mapping.size();
        if (first_mapping_count == 0) {
            first_mapping_ptr = nullptr;
        }

        return this->set_difference(
            other,
            first_mapping_ptr,
            first_mapping_count
        );
    }

    /// Select entries in these `Labels` that match the `selection`.
    ///
    /// The selection's names must be a subset of the names of these labels.
    ///
    /// All entries in these `Labels` that match one of the entry in the
    /// `selection` for all the selection's dimension will be picked. Any entry
    /// in the `selection` but not in these `Labels` will be ignored.
    ///
    /// @param selection definition of the selection criteria. Multiple entries
    ///        are interpreted as a logical `or` operation.
    /// @param selected on input, a pointer to an array with space for
    ///        `*selected_count` entries. On output, the first `*selected_count`
    ///        values will contain the index in `labels` of selected entries.
    /// @param selected_count on input, size of the `selected` array. On output,
    ///        this will contain the number of selected entries.
    void select(const Labels& selection, int64_t* selected, size_t *selected_count) const {
        details::check_status(mts_labels_select(
            labels_,
            selection.labels_,
            selected,
            selected_count
        ));
    }

    /// Select entries in these `Labels` that match the `selection`.
    ///
    /// This function does the same thing as the one above, but allocates and
    /// return the list of selected indexes in a `std::vector`
    std::vector<int64_t> select(const Labels& selection) const {
        auto selected_count = this->count();
        auto selected = std::vector<int64_t>(selected_count, -1);

        this->select(selection, selected.data(), &selected_count);

        selected.resize(selected_count);
        return selected;
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load previously saved ``Labels`` from the given path.
     *
     * This is identical to :cpp:func:`metatensor::io::load_labels`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    static Labels load(const std::string& path) {
        return metatensor::io::load_labels(path);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load previously saved ``Labels`` from a in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::load_labels_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    static Labels load_buffer(const uint8_t* buffer, size_t buffer_count) {
        return metatensor::io::load_labels_buffer(buffer, buffer_count);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load previously saved ``Labels`` from a in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::load_labels_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    template <typename Buffer>
    static Labels load_buffer(const Buffer& buffer) {
        return metatensor::io::load_labels_buffer<Buffer>(buffer);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Save ``Labels`` to the given path.
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
     * Save ``Labels`` to an in-memory buffer.
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
     * Save ``Labels`` to an in-memory buffer.
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

private:
    explicit Labels(): values_(static_cast<const int32_t*>(nullptr), {0, 0})
    {
        std::memset(&labels_, 0, sizeof(labels_));
    }

    explicit Labels(mts_labels_t labels):
        values_(labels.values, {labels.count, labels.size}),
        labels_(labels)
    {
        assert(labels_.internal_ptr_ != nullptr);

        for (size_t i=0; i<labels_.size; i++) {
            names_.push_back(labels_.names[i]);
        }
    }

    // the constructor below is ambiguous with the public constructor taking
    // `std::initializer_list`, so we use a private dummy struct argument to
    // remove the ambiguity.
    struct InternalConstructor {};
    Labels(const std::vector<std::string>& names, const NDArray<int32_t>& values, InternalConstructor):
        Labels(names, values.data(), values.shape()[0]) {}

    Labels(const std::vector<std::string>& names, const NDArray<int32_t>& values, assume_unique, InternalConstructor):
        Labels(names, values.data(), values.shape()[0], assume_unique{}) {}

    friend Labels details::labels_from_cxx(const std::vector<std::string>& names, const int32_t* values, size_t count, bool assume_unique);
    friend Labels io::load_labels(const std::string &path);
    friend Labels io::load_labels_buffer(const uint8_t* buffer, size_t buffer_count);
    friend class TensorMap;
    friend class TensorBlock;

    friend class metatensor_torch::LabelsHolder;

    std::vector<const char*> names_;
    NDArray<int32_t> values_;
    mts_labels_t labels_;

    friend bool operator==(const Labels& lhs, const Labels& rhs);
};

/// It is possible to store some user-provided data inside `Labels`, and access
/// it later. This class is used to take ownership of the data and corresponding
/// delete function before giving the data to metatensor.
///
/// User data inside `Labels` is an advanced functionality, that most users
/// should not need to interact with.
class LabelsUserData {
public:
    /// Create `LabelsUserData` containing the given `data`.
    ///
    /// `deleter` will be called when the data is dropped, and should
    /// free the corresponding memory.
    LabelsUserData(void* data, void(*deleter)(void*)): data_(data), deleter_(deleter) {}

    ~LabelsUserData() {
        if (deleter_ !=  nullptr) {
            deleter_(data_);
        }
    }

    /// LabelsUserData is not copy-constructible
    LabelsUserData(const LabelsUserData& other) = delete;
    /// LabelsUserData can not be copy-assigned
    LabelsUserData& operator=(const LabelsUserData& other) = delete;

    /// LabelsUserData is move-constructible
    LabelsUserData(LabelsUserData&& other) noexcept: LabelsUserData(nullptr, nullptr) {
        *this = std::move(other);
    }

    /// LabelsUserData be move-assigned
    LabelsUserData& operator=(LabelsUserData&& other) noexcept {
        if (deleter_ !=  nullptr) {
            deleter_(data_);
        }

        data_ = other.data_;
        deleter_ = other.deleter_;

        other.data_ = nullptr;
        other.deleter_ = nullptr;

        return *this;
    }

private:
    friend class Labels;

    void* data_;
    void(*deleter_)(void*);
};

inline void Labels::set_user_data(LabelsUserData user_data) {
    assert(labels_.internal_ptr_ != nullptr);

    details::check_status(mts_labels_set_user_data(
        labels_,
        user_data.data_,
        user_data.deleter_
    ));

    // the user data was moved inside `labels_`
    user_data.data_ = nullptr;
    user_data.deleter_ = nullptr;
}

/// Two Labels compare equal only if they have the same names and values in the
/// same order.
inline bool operator==(const Labels& lhs, const Labels& rhs) {
    if (lhs.names_.size() != rhs.names_.size()) {
        return false;
    }

    for (size_t i=0; i<lhs.names_.size(); i++) {
        if (std::strcmp(lhs.names_[i], rhs.names_[i]) != 0) {
            return false;
        }
    }

    return lhs.values() == rhs.values();
}

/// Two Labels compare equal only if they have the same names and values in the
/// same order.
inline bool operator!=(const Labels& lhs, const Labels& rhs) {
    return !(lhs == rhs);
}

namespace details {
    inline metatensor::Labels labels_from_cxx(
        const std::vector<std::string>& names,
        const int32_t* values,
        size_t count,
        bool assume_unique = false
    ) {
        mts_labels_t labels{};

        auto c_names = std::vector<const char*>();
        c_names.reserve(names.size());
        for (const auto& name: names) {
            c_names.push_back(name.c_str());
        }

        labels.names = c_names.data();
        labels.size = c_names.size();
        labels.count = count;
        labels.values = values;

        if (assume_unique) {
            details::check_status(mts_labels_create_assume_unique(&labels));
        } else {
            details::check_status(mts_labels_create(&labels));
        }

        return metatensor::Labels(labels);
    }
} // namespace details
} // namespace metatensor
