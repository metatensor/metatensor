#pragma once
#ifndef METATENSOR_HPP
#define METATENSOR_HPP

#include <array>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "metatensor.h"

#include "cxx/details/Error.hpp"

namespace metatensor_torch {
    class LabelsHolder;
    class TensorBlockHolder;
    class TensorMapHolder;
}

namespace metatensor {
class Labels;
class TensorBlock;
class TensorMap;

/// Tag for the creation of Labels without uniqueness checks
struct assume_unique {};

// Foundational utilities and forward declarations
namespace details {
    #include "cxx/details/details.hpp"
}
#include "cxx/io/save_load.hpp"

// The NDArray is a basic data container used by Labels
#include "cxx/data/ndarray.hpp"

// Labels are fundamental metadata, used by blocks and maps
#include "cxx/labels/labels.hpp"

// DataArrayBase and its implementations are needed by TensorBlock
#include "cxx/data/array_base.hpp"
#include "cxx/data/simple_data.hpp"
#include "cxx/data/empty_data.hpp"


// TensorBlock is ideally the lowest user-facing component
#include "cxx/tensor/tensorblock.hpp"

// TensorMap is the top-level container
#include "cxx/tensor/tensormap.hpp"

// I/O functions operate on all the above types
#include "cxx/io/save_load.hpp"

// Implementations need to be visible after all classes are defined
#include "cxx/details/default_create_array.inl"
namespace io {
    #include "cxx/io/save_load_impl.inl"
}

} // namespace metatensor

#endif /* METATENSOR_HPP */
