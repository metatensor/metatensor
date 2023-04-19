#ifndef EQUISTORE_TORCH_ARRAY_HPP
#define EQUISTORE_TORCH_ARRAY_HPP

#include <vector>

#include <torch/script.h>

#include <equistore.hpp>
#include "equistore/torch/exports.h"

namespace equistore_torch {

/// Equistore data origin for torch arrays. This is either 0 if no torch::Tensor
/// has been registered with equistore yet, or the origin for torch::Tensor.
extern eqs_data_origin_t TORCH_DATA_ORIGIN;

/// An `equistore::DataArrayBase` implementation using `torch::Tensor` to store
/// the data
class EQUISTORE_TORCH_EXPORT TorchDataArray: public equistore::DataArrayBase {
public:
    TorchDataArray(torch::Tensor tensor);

    virtual ~TorchDataArray() override = default;

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

    /*========================================================================*/
    /*          Functions to implement equistore::DataArrayBase               */
    /*========================================================================*/

    eqs_data_origin_t origin() const override;

    std::unique_ptr<equistore::DataArrayBase> copy() const override;

    std::unique_ptr<equistore::DataArrayBase> create(std::vector<uintptr_t> shape) const override;

    double* data() override;

    const std::vector<uintptr_t>& shape() const override;

    void reshape(std::vector<uintptr_t> shape) override;

    void swap_axes(uintptr_t axis_1, uintptr_t axis_2) override;

    void move_samples_from(
        const equistore::DataArrayBase& input,
        std::vector<eqs_sample_mapping_t> samples,
        uintptr_t property_start,
        uintptr_t property_end
    ) override;

private:
    // the actual data
    torch::Tensor tensor_;
};

}

#endif
