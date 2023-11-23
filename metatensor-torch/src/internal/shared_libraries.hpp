#ifndef METATENSOR_TORCH_SHARED_LIBRARIES_HPP
#define METATENSOR_TORCH_SHARED_LIBRARIES_HPP

#include <vector>
#include <string>

namespace metatensor_torch::details {

/// Get the full list of shared libraries loaded by the current module. Shared
/// libraries are returned with their path.
std::vector<std::string> get_loaded_libraries();

}


#endif
