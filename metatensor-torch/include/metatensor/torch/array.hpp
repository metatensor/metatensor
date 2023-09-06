#ifndef METATENSOR_TORCH_ARRAY_HPP
#define METATENSOR_TORCH_ARRAY_HPP

#include <vector>

#include <torch/script.h>

#include <metatensor.hpp>
#include "metatensor/torch/exports.h"

namespace metatensor_torch {

/// Metatensor data origin for torch arrays. This is either 0 if no torch::Tensor
/// has been registered with metatensor yet, or the origin for torch::Tensor.
extern mts_data_origin_t TORCH_DATA_ORIGIN;

/// An `metatensor::DataArrayBase` implementation using `torch::Tensor` to store
/// the data
class METATENSOR_TORCH_EXPORT TorchDataArray: public metatensor::DataArrayBase {
public:
    /// Create a `TorchDataArray` containing the given `tensor`
    TorchDataArray(torch::Tensor tensor);

    ~TorchDataArray() override = default;

    /// TorchDataArray can be copy-constructed
    TorchDataArray(const TorchDataArray&) = default;
    /// TorchDataArray can be copy-assigned
    TorchDataArray& operator=(const TorchDataArray&) = default;
    /// TorchDataArray can be move-constructed
    TorchDataArray(TorchDataArray&&) noexcept = default;
    /// TorchDataArray can be move-assigned
    TorchDataArray& operator=(TorchDataArray&&) noexcept = default;

    /// Get the underlying tensor
    torch::Tensor tensor() {
        return tensor_;
    }

    /// Get the underlying tensor
    const torch::Tensor& tensor() const {
        return tensor_;
    }

    /*========================================================================*/
    /*          Functions to implement metatensor::DataArrayBase               */
    /*========================================================================*/

    mts_data_origin_t origin() const override;

    std::unique_ptr<metatensor::DataArrayBase> copy() const override;

    std::unique_ptr<metatensor::DataArrayBase> create(std::vector<uintptr_t> shape) const override;

    double* data() & override;

    const std::vector<uintptr_t>& shape() const & override;

    void reshape(std::vector<uintptr_t> shape) override;

    void swap_axes(uintptr_t axis_1, uintptr_t axis_2) override;

    void move_samples_from(
        const metatensor::DataArrayBase& input,
        std::vector<mts_sample_mapping_t> samples,
        uintptr_t property_start,
        uintptr_t property_end
    ) override;

private:
    // cache the array shape as a vector of unsigned integers (as expected by
    // metatensor) instead of signed integer (as stored in torch::Tensor::sizes)
    std::vector<uintptr_t> shape_;
    void update_shape();

    // the actual data
    torch::Tensor tensor_;
};

}

#endif
