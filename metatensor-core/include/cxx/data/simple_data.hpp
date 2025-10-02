#pragma once
#include <array>
#include <cassert>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace metatensor {
/// Very basic implementation of DataArrayBase in C++.
///
/// This is included as an example implementation of DataArrayBase, and to make
/// metatensor usable without additional dependencies. For other uses cases, it
/// might be better to implement DataArrayBase on your data, using
/// functionalities from `Eigen`, `Boost.Array`, etc.
class SimpleDataArray: public metatensor::DataArrayBase {
public:
    /// Create a SimpleDataArray with the given `shape`, and all elements set to
    /// `value`
    SimpleDataArray(std::vector<uintptr_t> shape, double value = 0.0):
        shape_(std::move(shape)), data_(details::product(shape_), value) {}

    /// Create a SimpleDataArray with the given `shape` and `data`.
    ///
    /// The data is interpreted as a row-major n-dimensional array.
    SimpleDataArray(std::vector<uintptr_t> shape, std::vector<double> data):
        shape_(std::move(shape)),
        data_(std::move(data))
    {
        if (data_.size() != details::product(shape_)) {
            throw Error("the shape and size of the data don't match in SimpleDataArray");
        }
    }

    ~SimpleDataArray() override = default;

    /// SimpleDataArray can be copy-constructed
    SimpleDataArray(const SimpleDataArray&) = default;
    /// SimpleDataArray can be copy-assigned
    SimpleDataArray& operator=(const SimpleDataArray&) = default;
    /// SimpleDataArray can be move-constructed
    SimpleDataArray(SimpleDataArray&&) noexcept = default;
    /// SimpleDataArray can be move-assigned
    SimpleDataArray& operator=(SimpleDataArray&&) noexcept = default;

    mts_data_origin_t origin() const override {
        mts_data_origin_t origin = 0;
        mts_register_data_origin("metatensor::SimpleDataArray", &origin);
        return origin;
    }

    double* data() & override {
        return data_.data();
    }

    const std::vector<uintptr_t>& shape() const & override {
        return shape_;
    }

    void reshape(std::vector<uintptr_t> shape) override {
        if (details::product(shape_) != details::product(shape)) {
            throw metatensor::Error("invalid shape in reshape");
        }
        shape_ = std::move(shape);
    }

    void swap_axes(uintptr_t axis_1, uintptr_t axis_2) override {
        auto new_data = std::vector<double>(details::product(shape_), 0.0);
        auto new_shape = shape_;
        std::swap(new_shape[axis_1], new_shape[axis_2]);

        for (size_t i=0; i<details::product(shape_); i++) {
            auto index = details::cartesian_index(shape_, i);
            std::swap(index[axis_1], index[axis_2]);

            new_data[details::linear_index(new_shape, index)] = data_[i];
        }

        shape_ = std::move(new_shape);
        data_ = std::move(new_data);
    }

    std::unique_ptr<DataArrayBase> copy() const override {
        return std::unique_ptr<DataArrayBase>(new SimpleDataArray(*this));
    }

    std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape) const override {
        return std::unique_ptr<DataArrayBase>(new SimpleDataArray(std::move(shape)));
    }

    void move_samples_from(
        const DataArrayBase& input,
        std::vector<mts_sample_mapping_t> samples,
        uintptr_t property_start,
        uintptr_t property_end
    ) override {
        const auto& input_array = dynamic_cast<const SimpleDataArray&>(input);
        assert(input_array.shape_.size() == this->shape_.size());

        size_t property_count = property_end - property_start;
        size_t property_dim = shape_.size() - 1;
        assert(input_array.shape_[property_dim] == property_count);

        auto input_index = std::vector<size_t>(shape_.size(), 0);
        auto output_index = std::vector<size_t>(shape_.size(), 0);

        for (const auto& sample: samples) {
            input_index[0] = sample.input;
            output_index[0] = sample.output;

            if (property_dim == 1) {
                // no components
                for (size_t property_i=0; property_i<property_count; property_i++) {
                    input_index[property_dim] = property_i;
                    output_index[property_dim] = property_i + property_start;

                    auto value = input_array.data_[details::linear_index(input_array.shape_, input_index)];
                    this->data_[details::linear_index(shape_, output_index)] = value;
                }
            } else {
                auto last_component_dim = shape_.size() - 2;
                for (size_t component_i=1; component_i<shape_.size() - 1; component_i++) {
                    input_index[component_i] = 0;
                }

                bool done = false;
                while (!done) {
                    for (size_t component_i=1; component_i<shape_.size() - 1; component_i++) {
                        output_index[component_i] = input_index[component_i];
                    }

                    for (size_t property_i=0; property_i<property_count; property_i++) {
                        input_index[property_dim] = property_i;
                        output_index[property_dim] = property_i + property_start;

                        auto value = input_array.data_[details::linear_index(input_array.shape_, input_index)];
                        this->data_[details::linear_index(shape_, output_index)] = value;
                    }

                    input_index[last_component_dim] += 1;
                    for (size_t component_i=last_component_dim; component_i>2; component_i--) {
                        if (input_index[component_i] >= shape_[component_i]) {
                            input_index[component_i] = 0;
                            input_index[component_i - 1] += 1;
                        }
                    }

                    if (input_index[1] >= shape_[1]) {
                        done = true;
                    }
                }
            }
        }
    }

    /// Get a const view of the data managed by this SimpleDataArray
    NDArray<double> view() const {
        return NDArray<double>(data_.data(), shape_);
    }

    /// Get a mutable view of the data managed by this SimpleDataArray
    NDArray<double> view() {
        return NDArray<double>(data_.data(), shape_);
    }

    /// Extract a reference to SimpleDataArray out of an `mts_array_t`.
    ///
    /// This function fails if the `mts_array_t` does not contain a
    /// SimpleDataArray.
    static SimpleDataArray& from_mts_array(mts_array_t& array) {
        mts_data_origin_t origin = 0;
        auto status = array.origin(array.ptr, &origin);
        if (status != MTS_SUCCESS) {
            throw Error("failed to get data origin");
        }

        std::array<char, 64> buffer = {0};
        status = mts_get_data_origin(origin, buffer.data(), buffer.size());
        if (status != MTS_SUCCESS || std::string(buffer.data()) != "metatensor::SimpleDataArray") {
            throw Error("this array is not a metatensor::SimpleDataArray");
        }

        auto* base = static_cast<DataArrayBase*>(array.ptr);
        return dynamic_cast<SimpleDataArray&>(*base);
    }

    /// Extract a const reference to SimpleDataArray out of an `mts_array_t`.
    ///
    /// This function fails if the `mts_array_t` does not contain a
    /// SimpleDataArray.
    static const SimpleDataArray& from_mts_array(const mts_array_t& array) {
        mts_data_origin_t origin = 0;
        auto status = array.origin(array.ptr, &origin);
        if (status != MTS_SUCCESS) {
            throw Error("failed to get data origin");
        }

        std::array<char, 64> buffer = {0};
        status = mts_get_data_origin(origin, buffer.data(), buffer.size());
        if (status != MTS_SUCCESS || std::string(buffer.data()) != "metatensor::SimpleDataArray") {
            throw Error("this array is not a metatensor::SimpleDataArray");
        }

        const auto* base = static_cast<const DataArrayBase*>(array.ptr);
        return dynamic_cast<const SimpleDataArray&>(*base);
    }

private:
    std::vector<uintptr_t> shape_;
    std::vector<double> data_;

    friend bool operator==(const SimpleDataArray& lhs, const SimpleDataArray& rhs);
};

/// Two SimpleDataArray compare as equal if they have the exact same shape and
/// data.
inline bool operator==(const SimpleDataArray& lhs, const SimpleDataArray& rhs) {
    return lhs.shape_ == rhs.shape_ && lhs.data_ == rhs.data_;
}

/// Two SimpleDataArray compare as equal if they have the exact same shape and
/// data.
inline bool operator!=(const SimpleDataArray& lhs, const SimpleDataArray& rhs) {
    return !(lhs == rhs);
}
} // namespace metatensor
