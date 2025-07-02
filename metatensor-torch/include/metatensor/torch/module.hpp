#ifndef METATENSOR_TORCH_MODULE_HPP
#define METATENSOR_TORCH_MODULE_HPP

#include <torch/script.h>

#include "metatensor/torch/exports.h"

namespace metatensor_torch {

/// Replacement class for `torch::jit::Module` to be used when the module
/// contains metatensor data. This class overrides the behavior of `to()` to
/// also move metatensor data to the correct dtype and device.
class METATENSOR_TORCH_EXPORT Module: public torch::jit::Module {
public:
    /// Construct a `metatensor_torch::Module` wrapping the given
    /// `torch::jit::Module`.
    Module(torch::jit::Module module): torch::jit::Module(std::move(module)) {}

    /// Move all the data in the module to the given `device` and `dtype`
    void to(at::Device device, at::ScalarType dtype, bool non_blocking = false);

    /// Move all the data in the module to the given `dtype`
    void to(at::ScalarType dtype, bool non_blocking = false);

    /// Move all the data in the module to the given `device`
    void to(at::Device device, bool non_blocking = false);

private:
    void to_impl_(
        const torch::optional<at::Device>& device,
        const torch::optional<at::ScalarType>& dtype,
        bool non_blocking
    );
};

}

#endif
