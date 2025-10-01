#pragma once
/// Compute the product of all values in the `shape` vector
inline size_t product(const std::vector<size_t>& shape) {
    size_t result = 1;
    for (auto size: shape) {
        result *= size;
    }
    return result;
}

/// Get the N-dimensional index corresponding to the given linear `index`
/// and array `shape`
inline std::vector<size_t> cartesian_index(const std::vector<size_t>& shape, size_t index) {
    auto result = std::vector<size_t>(shape.size(), 0);
    for (size_t i=0; i<shape.size(); i++) {
        result[i] = index % shape[i];
        index = index / shape[i];
    }
    assert(index == 0);
    return result;
}

/// Get the linear index corresponding to the N-dimensional
/// `index[index_size]`, according to the given array `shape`
inline size_t linear_index(const std::vector<size_t>& shape, const size_t* index, size_t index_size) {
    assert(index_size != 0);
    assert(index_size == shape.size());

    if (index_size == 1) {
        assert(index[0] < shape[0] && "out of bounds");
        return index[0];
    } else {
        assert(index[0] < shape[0]);
        auto linear_index = index[0];
        for (size_t i=1; i<index_size; i++) {
            assert(index[i] < shape[i] && "out of bounds");
            linear_index *= shape[i];
            linear_index += index[i];
        }

        return linear_index;
    }
}

template<size_t N>
size_t linear_index(const std::vector<size_t>& shape, const std::array<size_t, N>& index) {
    return linear_index(shape, index.data(), index.size());
}

inline size_t linear_index(const std::vector<size_t>& shape, const std::vector<size_t>& index) {
    return linear_index(shape, index.data(), index.size());
}

Labels labels_from_cxx(const std::vector<std::string>& names, const int32_t* values, size_t count, bool assume_unique);

mts_status_t default_create_array(
    const uintptr_t* shape_ptr,
    uintptr_t shape_count,
    mts_array_t* array
);
