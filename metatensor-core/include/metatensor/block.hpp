#pragma once

#include <cassert>
#include <cstring>
#include <string>
#include <vector>

#include <metatensor.h>

#include "./errors.hpp"
#include "./arrays.hpp"
#include "./labels.hpp"
#include "./io_fwd.hpp"

namespace metatensor_torch {
    class TensorBlockHolder;
}

namespace metatensor {
/// Basic building block for a tensor map.
///
/// A single block contains a n-dimensional `mts_array_t` (or `DataArrayBase`),
/// and n sets of `Labels` (one for each dimension). The first dimension is the
/// *samples* dimension, the last dimension is the *properties* dimension. Any
/// intermediate dimension is called a *component* dimension.
///
/// Samples should be used to describe *what* we are representing, while
/// properties should contain information about *how* we are representing it.
/// Finally, components should be used to describe vectorial or tensorial
/// components of the data.
///
/// A block can also contain gradients of the values with respect to a variety
/// of parameters. In this case, each gradient has a separate set of samples,
/// and possibly components but share the same property labels as the values.
class TensorBlock final {
public:
    /// Create a new TensorBlock containing the given `values` array.
    ///
    /// The different dimensions of the values are described by `samples`,
    /// `components` and `properties` `Labels`
    TensorBlock(
        std::unique_ptr<DataArrayBase> values,
        const Labels& samples,
        const std::vector<Labels>& components,
        const Labels& properties
    ):
        block_(nullptr),
        is_view_(false)
    {
        auto c_components = std::vector<mts_labels_t>();
        for (const auto& component: components) {
            c_components.push_back(component.as_mts_labels_t());
        }
        block_ = mts_block(
            DataArrayBase::to_mts_array_t(std::move(values)),
            samples.as_mts_labels_t(),
            c_components.data(),
            c_components.size(),
            properties.as_mts_labels_t()
        );

        details::check_pointer(block_);
    }

    ~TensorBlock() {
        if (!is_view_) {
            mts_block_free(block_);
        }
    }

    /// TensorBlock can NOT be copy constructed, use TensorBlock::clone instead
    TensorBlock(const TensorBlock&) = delete;

    /// TensorBlock can NOT be copy assigned, use TensorBlock::clone instead
    TensorBlock& operator=(const TensorBlock& other) = delete;

    /// TensorBlock can be move constructed
    TensorBlock(TensorBlock&& other) noexcept : TensorBlock() {
        *this = std::move(other);
    }

    /// TensorBlock can be moved assigned
    TensorBlock& operator=(TensorBlock&& other) noexcept {
        if (!is_view_) {
            mts_block_free(block_);
        }

        this->block_ = other.block_;
        this->is_view_ = other.is_view_;
        other.block_ = nullptr;
        other.is_view_ = true;

        return *this;
    }

    /// Make a copy of this `TensorBlock`, including all the data contained inside
    TensorBlock clone() const {
        auto copy = TensorBlock();
        copy.is_view_ = false;
        copy.block_ = mts_block_copy(this->block_);
        details::check_pointer(copy.block_);
        return copy;
    }

    /// Get a copy of the metadata in this block (i.e. samples, components, and
    /// properties), ignoring the data itself.
    ///
    /// The resulting block values will be an `EmptyDataArray` instance, which
    /// does not contain any data.
    TensorBlock clone_metadata_only() const {
        auto block = TensorBlock(
            std::unique_ptr<EmptyDataArray>(new EmptyDataArray(this->values_shape())),
            this->samples(),
            this->components(),
            this->properties()
        );

        for (const auto& parameter: this->gradients_list()) {
            auto gradient = this->gradient(parameter);
            block.add_gradient(parameter, gradient.clone_metadata_only());
        }

        return block;
    }

    /// Get a view in the values in this block
    NDArray<double> values() & {
        auto array = this->mts_array();
        double* data = nullptr;
        details::check_status(array.data(array.ptr, &data));

        return NDArray<double>(data, this->values_shape());
    }

    NDArray<double> values() && = delete;

    /// Access the sample `Labels` for this block.
    ///
    /// The entries in these labels describe the first dimension of the
    /// `values()` array.
    Labels samples() const {
        return this->labels(0);
    }

    /// Access the component `Labels` for this block.
    ///
    /// The entries in these labels describe intermediate dimensions of the
    /// `values()` array.
    std::vector<Labels> components() const {
        auto shape = this->values_shape();

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
    /// a block and all of its gradients.
    Labels properties() const {
        auto shape = this->values_shape();
        return this->labels(shape.size() - 1);
    }

    /// Add a set of gradients with respect to `parameters` in this block.
    ///
    /// @param parameter add gradients with respect to this `parameter` (e.g.
    ///                 `"positions"`, `"cell"`, ...)
    /// @param gradient a `TensorBlock` whose values contain the gradients with
    ///                 respect to the `parameter`. The labels of the gradient
    ///                 `TensorBlock` should be organized as follows: its
    ///                 `samples` must contain `"sample"` as the first label,
    ///                 which establishes a correspondence with the `samples` of
    ///                 the original `TensorBlock`; its components must contain
    ///                 at least the same components as the original
    ///                 `TensorBlock`, with any additional component coming
    ///                 before those; its properties must match those of the
    ///                 original `TensorBlock`.
    void add_gradient(const std::string& parameter, TensorBlock gradient) {
        if (is_view_) {
            throw Error(
                "can not call TensorBlock::add_gradient on this block since "
                "it is a view inside a TensorMap"
            );
        }

        details::check_status(mts_block_add_gradient(
            block_,
            parameter.c_str(),
            gradient.release()
        ));
    }

    /// Get a list of all gradients defined in this block.
    std::vector<std::string> gradients_list() const {
        const char*const * parameters = nullptr;
        uintptr_t count = 0;
        details::check_status(mts_block_gradients_list(
            block_,
            &parameters,
            &count
        ));

        auto result = std::vector<std::string>();
        for (uint64_t i=0; i<count; i++) {
            result.emplace_back(parameters[i]);
        }

        return result;
    }

    /// Get the gradient in this block with respect to the given `parameter`.
    /// The gradient is returned as a TensorBlock itself.
    ///
    /// @param parameter check for gradients with respect to this `parameter`
    ///                  (e.g. `"positions"`, `"cell"`, ...)
    TensorBlock gradient(const std::string& parameter) const {
        mts_block_t* gradient_block = nullptr;
        details::check_status(
            mts_block_gradient(block_, parameter.c_str(), &gradient_block)
        );
        details::check_pointer(gradient_block);
        return TensorBlock::unsafe_view_from_ptr(gradient_block);
    }

    /// Get the `mts_block_t` pointer corresponding to this block.
    ///
    /// The block pointer is still managed by the current `TensorBlock`
    mts_block_t* as_mts_block_t() & {
        if (is_view_) {
            throw Error(
                "can not call non-const TensorBlock::as_mts_block_t on this "
                "block since it is a view inside a TensorMap"
            );
        }
        return block_;
    }

    /// const version of `as_mts_block_t`
    const mts_block_t* as_mts_block_t() const & {
        return block_;
    }

    const mts_block_t* as_mts_block_t() && = delete;

    /// Create a new TensorBlock taking ownership of a raw `mts_block_t` pointer.
    static TensorBlock unsafe_from_ptr(mts_block_t* ptr) {
        auto block = TensorBlock();
        block.block_ = ptr;
        block.is_view_ = false;
        return block;
    }

    /// Create a new TensorBlock which is a view corresponding to a raw
    /// `mts_block_t` pointer.
    static TensorBlock unsafe_view_from_ptr(mts_block_t* ptr) {
        auto block = TensorBlock();
        block.block_ = ptr;
        block.is_view_ = true;
        return block;
    }

    /// Get a raw `mts_array_t` corresponding to the values in this block.
    mts_array_t mts_array() {
        mts_array_t array;
        std::memset(&array, 0, sizeof(array));

        details::check_status(
            mts_block_data(block_, &array)
        );
        return array;
    }

    /// Get the labels in this block associated with the given `axis`.
    Labels labels(uintptr_t axis) const {
        mts_labels_t labels;
        std::memset(&labels, 0, sizeof(labels));
        details::check_status(mts_block_labels(
            block_, axis, &labels
        ));

        return Labels(labels);
    }

    /// Get the shape of the value array for this block
    std::vector<uintptr_t> values_shape() const {
        auto array = this->const_mts_array();

        const uintptr_t* shape = nullptr;
        uintptr_t shape_count = 0;
        details::check_status(array.shape(array.ptr, &shape, &shape_count));
        assert(shape_count >= 2);

        return {shape, shape + shape_count};
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load a previously saved ``TensorBlock`` from the given path.
     *
     * This is identical to :cpp:func:`metatensor::io::load_block`, and provided
     * as a convenience API.
     *
     * \endverbatim
     */
    static TensorBlock load(
        const std::string& path,
        mts_create_array_callback_t create_array = details::default_create_array
    ) {
        return metatensor::io::load_block(path, create_array);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load a previously saved ``TensorBlock`` from a in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::load_block_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    static TensorBlock load_buffer(
        const uint8_t* buffer,
        size_t buffer_count,
        mts_create_array_callback_t create_array = details::default_create_array
    ) {
        return metatensor::io::load_block_buffer(buffer, buffer_count, create_array);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load a previously saved ``TensorBlock`` from a in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::load_block_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    template <typename Buffer>
    static TensorBlock load_buffer(
        const Buffer& buffer,
        mts_create_array_callback_t create_array = details::default_create_array
    ) {
        return metatensor::io::load_block_buffer<Buffer>(buffer, create_array);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Save this ``TensorBlock`` to the given path.
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
     * Save this ``TensorBlock`` to an in-memory buffer.
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
     * Save this ``TensorBlock`` to an in-memory buffer.
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
    /// Constructor of a TensorBlock not associated with anything
    explicit TensorBlock(): block_(nullptr), is_view_(true) {}

    /// Create a C++ TensorBlock from a C `mts_block_t` pointer. The C++
    /// block takes ownership of the C pointer.
    explicit TensorBlock(mts_block_t* block): block_(block), is_view_(false) {}

    /// Get the `mts_array_t` for this block.
    ///
    /// The returned `mts_array_t` should only be used in a const context
    mts_array_t const_mts_array() const {
        mts_array_t array;
        std::memset(&array, 0, sizeof(array));

        details::check_status(
            mts_block_data(block_, &array)
        );
        return array;
    }

    /// Release the `mts_block_t` pointer corresponding to this `TensorBlock`.
    ///
    /// The block pointer is **no longer** managed by the current `TensorBlock`,
    /// and should manually be freed when no longer required.
    mts_block_t* release() {
         if (is_view_) {
            throw Error(
                "can not call TensorBlock::release on this "
                "block since it is a view inside a TensorMap"
            );
        }
        auto* ptr = block_;
        block_ = nullptr;
        is_view_ = false;
        return ptr;
    }

    friend class TensorMap;
    friend class metatensor_torch::TensorBlockHolder;
    friend TensorBlock metatensor::io::load_block(
        const std::string& path,
        mts_create_array_callback_t create_array
    );
    friend TensorBlock metatensor::io::load_block_buffer(
        const uint8_t* buffer,
        size_t buffer_count,
        mts_create_array_callback_t create_array
    );

    mts_block_t* block_;
    bool is_view_;
};
} // namespace metatensor
