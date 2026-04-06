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
    /// Create Labels from the given dimension names and values.
    ///
    /// The Labels take ownership of the values.
    Labels(const std::vector<std::string>& dimensions, MtsArray values):
        labels_(nullptr)
    {
        auto c_dimensions = std::vector<const char*>();
        c_dimensions.reserve(dimensions.size());
        for (const auto& name: dimensions) {
            c_dimensions.push_back(name.c_str());
        }

        labels_ = mts_labels(
            c_dimensions.data(), c_dimensions.size(), std::move(values).release()
        );
        if (labels_ == nullptr) {
            throw Error(mts_last_error());
        }
    }

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
        const std::vector<std::string>& dimensions,
        std::initializer_list<std::initializer_list<int32_t>> values
    ): Labels(dimensions, NDArray<int32_t>(values, dimensions.size()), InternalConstructor{}) {}

    /// Create Labels from the given dimension names and a backing mts_array_t,
    /// assuming uniqueness of entries (no uniqueness check is performed).
    ///
    /// The Labels take ownership of the array.
    Labels(const std::vector<std::string>& dimensions, MtsArray array, assume_unique):
        labels_(nullptr)
    {
        auto c_dimensions = std::vector<const char*>();
        c_dimensions.reserve(dimensions.size());
        for (const auto& name: dimensions) {
            c_dimensions.push_back(name.c_str());
        }

        labels_ = mts_labels_assume_unique(
            c_dimensions.data(), c_dimensions.size(), std::move(array).release()
        );
        if (labels_ == nullptr) {
            throw Error(mts_last_error());
        }
    }

    /// Create an empty set of Labels with the given dimension names.
    explicit Labels(const std::vector<std::string>& dimensions):
        Labels(dimensions, static_cast<const int32_t*>(nullptr), 0) {}

    /// Create labels with the given `dimensions` and `values`. `values` must be
    /// an array with `count x dimensions.size()` elements.
    Labels(const std::vector<std::string>& dimensions, const int32_t* values, size_t count):
        Labels(details::labels_from_cxx(dimensions, values, count, false)) {}

    /// Unchecked variant, caller promises the labels are unique. Calling with
    /// non-unique entries is invalid and can lead to crashes or infinite loops.
    Labels(const std::vector<std::string>& dimensions, const int32_t* values, size_t count, assume_unique):
        Labels(details::labels_from_cxx(dimensions, values, count, true)) {}

    ~Labels() {
        if (labels_ != nullptr) {
            mts_labels_free(labels_);
            labels_ = nullptr;
        }
    }

    /// Labels is copy-constructible
    Labels(const Labels& other): labels_(nullptr) {
        *this = other;
    }

    /// Labels can be copy-assigned
    Labels& operator=(const Labels& other) {
        if (this == &other) {
            return *this;
        }

        if (labels_ != nullptr) {
            mts_labels_free(labels_);
            labels_ = nullptr;
        }

        labels_ = mts_labels_clone(other.labels_);
        if (labels_ == nullptr) {
            throw Error(mts_last_error());
        }
        return *this;
    }

    /// Labels is move-constructible
    Labels(Labels&& other) noexcept: labels_(nullptr) {
        *this = std::move(other);
    }

    /// Labels can be move-assigned
    Labels& operator=(Labels&& other) noexcept {
        if (labels_ != nullptr) {
            mts_labels_free(labels_);
        }
        this->labels_ = other.labels_;
        other.labels_ = nullptr;
        return *this;
    }

    /// Get the names of the dimensions used in these `Labels`.
    std::vector<const char*> names() const {
        if (labels_ == nullptr) {
            return {};
        }

        const char* const* dimensions_ptr = nullptr;
        size_t dimensions_count = 0;
        details::check_status(mts_labels_dimensions(labels_, &dimensions_ptr, &dimensions_count));
        return std::vector<const char*>(dimensions_ptr, dimensions_ptr + dimensions_count);
    }

    /// Get the device of the values for these `Labels`.
    DLDevice device() const {
        auto array = this->mts_array();
        return array.device();
    }

    /// Get the number of entries in this set of Labels.
    size_t count() const {
        auto values = this->mts_array();
        return values.shape()[0];
    }

    /// Get the number of dimensions in this set of Labels.
    ///
    /// This is the same as `shape()[1]` for the corresponding values array
    size_t size() const {
        return this->names().size();
    }

    /// Get the underlying `mts_labels_t` pointer
    const mts_labels_t* as_mts_labels_t() const {
        return labels_;
    }

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
        assert(labels_ != nullptr);

        int64_t result = 0;
        details::check_status(mts_labels_position(labels_, entry, length, &result));
        return result;
    }

    /// Get the array of values for these Labels as an `mts_array_t`.
    MtsArray mts_array() const {
        mts_array_t array;
        std::memset(&array, 0, sizeof(array));
        details::check_status(mts_labels_values(labels_, &array));
        return MtsArray(array);
    }

    /// Get the array of values for these Labels as a DLPack array on the
    /// requested `device`.
    ///
    /// @param device the DLPack device to request data on (default: CPU)
    /// @param stream pointer to a device stream, or nullptr for default
    DLPackArray<int32_t> values(DLDevice device = {kDLCPU, 0}, const int64_t* stream = nullptr) const {
        auto mts_array = this->mts_array();

        return mts_array.as_dlpack_array<int32_t>(device, stream, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION});
    }

    /// Get the array of values for these Labels on CPU.
    ///
    /// This can trigger a copy if the values are not already on CPU, but
    /// following calls to this function will then return a view without
    /// copying.
    NDArray<int32_t> values_cpu() const {
        const int32_t* values = nullptr;
        uintptr_t count = 0;
        uintptr_t size = 0;
        details::check_status(mts_labels_values_cpu(labels_, &values, &count, &size));

        return NDArray<int32_t>(values, {count, size});
    }

    /// Take the union of these `Labels` with `other`.
    ///
    /// If requested, this function can also give the positions in the union
    /// where each entry of the input `Labels` ended up.
    ///
    /// The output data will be on CPU, regardless of the device of the inputs.
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
        const mts_labels_t* result = nullptr;

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
    /// The output data will be on CPU, regardless of the device of the inputs.
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
    /// The output data will be on CPU, regardless of the device of the inputs.
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
        const mts_labels_t* result = nullptr;

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
    /// The output data will be on CPU, regardless of the device of the inputs.
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
    /// The output data will be on CPU, regardless of the device of the inputs.
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
        const mts_labels_t* result = nullptr;

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
    /// The output data will be on CPU, regardless of the device of the inputs.
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
    /// Construct from an owning raw pointer (takes ownership)
    explicit Labels(const mts_labels_t* ptr): labels_(ptr) {
        assert(labels_ != nullptr);
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

    /// Owning pointer to the opaque labels
    const mts_labels_t* labels_;

    friend bool operator==(const Labels& lhs, const Labels& rhs);
};

/// Two Labels compare equal only if they have the same names and values in the
/// same order.
inline bool operator==(const Labels& lhs, const Labels& rhs) {
    auto lhs_names = lhs.names();
    auto rhs_names = rhs.names();
    if (lhs_names.size() != rhs_names.size()) {
        return false;
    }

    for (size_t i=0; i<lhs_names.size(); i++) {
        if (std::strcmp(lhs_names[i], rhs_names[i]) != 0) {
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
        auto c_names = std::vector<const char*>();
        c_names.reserve(names.size());
        for (const auto& name: names) {
            c_names.push_back(name.c_str());
        }

        // Wrap raw values in a SimpleDataArray<int32_t> to create an mts_array_t
        auto data = std::vector<int32_t>();
        size_t n_elements = count * names.size();
        if (values != nullptr && n_elements > 0) {
            data.assign(values, values + n_elements);
        }
        auto shape = std::vector<uintptr_t>{count, names.size()};
        auto cxx_array = std::unique_ptr<DataArrayBase>(
            new SimpleDataArray<int32_t>(std::move(shape), std::move(data))
        );
        auto array = DataArrayBase::to_mts_array(std::move(cxx_array));

        const mts_labels_t* labels;
        if (assume_unique) {
            labels = mts_labels_assume_unique(
                c_names.data(),
                c_names.size(),
                std::move(array).release()
            );
        } else {
            labels = mts_labels(
                c_names.data(),
                c_names.size(),
                std::move(array).release()
            );
        }

        details::check_pointer(labels);

        return metatensor::Labels(labels);
    }
} // namespace details
} // namespace metatensor
