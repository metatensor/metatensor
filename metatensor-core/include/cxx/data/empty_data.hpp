#pragma once
/// An implementation of `DataArrayBase` containing no data.
///
/// This class only tracks it's shape, and can be used when only the metadata
/// of a `TensorBlock` is important, leaving the data unspecified.
class EmptyDataArray: public metatensor::DataArrayBase {
public:
    /// Create ae `EmptyDataArray` with the given `shape`
    EmptyDataArray(std::vector<uintptr_t> shape):
        shape_(std::move(shape)) {}

    ~EmptyDataArray() override = default;

    /// EmptyDataArray can be copy-constructed
    EmptyDataArray(const EmptyDataArray&) = default;
    /// EmptyDataArray can be copy-assigned
    EmptyDataArray& operator=(const EmptyDataArray&) = default;
    /// EmptyDataArray can be move-constructed
    EmptyDataArray(EmptyDataArray&&) noexcept = default;
    /// EmptyDataArray can be move-assigned
    EmptyDataArray& operator=(EmptyDataArray&&) noexcept = default;

    mts_data_origin_t origin() const override {
        mts_data_origin_t origin = 0;
        mts_register_data_origin("metatensor::EmptyDataArray", &origin);
        return origin;
    }

    double* data() & override {
        throw metatensor::Error("can not call `data` for an EmptyDataArray");
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
        std::swap(shape_[axis_1], shape_[axis_2]);
    }

    std::unique_ptr<DataArrayBase> copy() const override {
        return std::unique_ptr<DataArrayBase>(new EmptyDataArray(*this));
    }

    std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape) const override {
        return std::unique_ptr<DataArrayBase>(new EmptyDataArray(std::move(shape)));
    }

    void move_samples_from(const DataArrayBase&, std::vector<mts_sample_mapping_t>, uintptr_t, uintptr_t) override {
        throw metatensor::Error("can not call `move_samples_from` for an EmptyDataArray");
    }

private:
    std::vector<uintptr_t> shape_;
};
