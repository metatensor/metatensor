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
    Module(torch::jit::Module module): torch::jit::Module(std::move(module)) {}

    void to(at::Device device, at::ScalarType dtype, bool non_blocking = false);
    void to(at::ScalarType dtype, bool non_blocking = false);
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
