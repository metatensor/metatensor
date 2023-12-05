#ifndef METATENSOR_TORCH_SCALAR_TYPE_NAME_HPP
#define METATENSOR_TORCH_SCALAR_TYPE_NAME_HPP

#include <torch/types.h>

namespace metatensor_torch {
namespace {

// torch::toString(scalar_type) returns different names compared to the Python
// representation of dtypes, so we use this custom function to get similar names
// as much as possible.
inline std::string scalar_type_name(torch::ScalarType scalar_type) {
    switch (scalar_type) {
        case torch::ScalarType::Byte:
        case torch::ScalarType::Char:
            return "torch.int8";
        case torch::ScalarType::Short:
            return "torch.int16";
        case torch::ScalarType::Int:
            return "torch.int32";
        case torch::ScalarType::Long:
            return "torch.int64";
        case torch::ScalarType::Half:
            return "torch.float16";
        case torch::ScalarType::Float:
            return "torch.float32";
        case torch::ScalarType::Double:
            return "torch.float64";
        case torch::ScalarType::ComplexHalf:
            return "torch.complex32";
        case torch::ScalarType::ComplexFloat:
            return "torch.complex64";
        case torch::ScalarType::ComplexDouble:
            return "torch.complex128";
        case torch::ScalarType::Bool:
            return "torch.bool";
        default:
            return torch::toString(scalar_type);
    }
}

#endif

}}
