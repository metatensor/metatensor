#ifndef METATENSOR_TORCH_SHARED_LIBRARIES_HPP
#define METATENSOR_TORCH_SHARED_LIBRARIES_HPP

#include <vector>
#include <string>

namespace metatensor_torch::details {

/// Get the full list of shared libraries loaded by the current executable.
/// This function returns the path to the libraries.
std::vector<std::string> get_loaded_libraries();

/// Try to load a shared library, trying the candidates in order, and if they
/// all fail trying to load by name. This function returns if it managed to find
/// and load the library.
bool load_library(const std::string& name, const std::vector<std::string>& candidates);

}


#endif
