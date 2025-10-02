#pragma once
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <vector>

namespace metatensor {
/// Simple N-dimensional array interface
///
/// This class can either be a non-owning view inside some existing memory (for
/// example memory allocated by Rust); or own its memory (in the form of an
/// `std::vector<double>`). If the array does not own its memory, accessing it
/// is only valid for as long as the memory is kept alive.
///
/// The API of this class is very intentionally minimal to keep metatensor as
/// simple as possible. Feel free to wrap the corresponding data inside types
/// with richer API such as Eigen, Boost, etc.
template<typename T>
class NDArray {
public:
    /// Create a new empty `NDArray`, with shape `[0, 0]`.
    NDArray(): NDArray(nullptr, {0, 0}, true) {}

    /// Create a new `NDArray` using a non-owning view in `const` memory with
    /// the given `shape`.
    ///
    /// `data` must point to contiguous memory containing the right number of
    /// elements as described by the `shape`, which will be interpreted as an
    /// N-dimensional array in row-major order. The resulting `NDArray` is
    /// only valid for as long as `data` is.
    NDArray(const T* data, std::vector<size_t> shape):
        NDArray(data, std::move(shape), /*is_const*/ true) {}

    /// Create a new `NDArray` using a non-owning view in non-`const` memory
    /// with the given `shape`.
    ///
    /// `data` must point to contiguous memory containing the right number of
    /// elements as described by the `shape`, which will be interpreted as an
    /// N-dimensional array in row-major order. The resulting `NDArray` is
    /// only valid for as long as `data` is.
    NDArray(T* data, std::vector<size_t> shape):
        NDArray(data, std::move(shape), /*is_const*/ false) {}

    /// Create a new `NDArray` *owning* its `data` with the given `shape`.
    NDArray(std::vector<T> data, std::vector<size_t> shape):
        NDArray(data.data(), std::move(shape), /*is_const*/ false)
    {
        using vector_t = std::vector<T>;

        owned_data_ = reinterpret_cast<void*>(new vector_t(std::move(data)));
        deleter_ = [](void* data){
            auto data_vector = reinterpret_cast<vector_t*>(data);
            delete data_vector;
        };
    }

    ~NDArray() noexcept {
        deleter_(this->owned_data_);
    }

    /// NDArray is not copy-constructible
    NDArray(const NDArray&) = delete;
    /// NDArray can not be copy-assigned
    NDArray& operator=(const NDArray& other) = delete;

    /// NDArray is move-constructible
    NDArray(NDArray&& other) noexcept: NDArray() {
        *this = std::move(other);
    }

    /// NDArray can be move-assigned
    NDArray& operator=(NDArray&& other) noexcept {
        this->deleter_(this->owned_data_);

        this->data_ = std::move(other.data_);
        this->shape_ = std::move(other.shape_);
        this->is_const_ = other.is_const_;
        this->owned_data_ = other.owned_data_;
        this->deleter_ = std::move(other.deleter_);

        other.data_ = nullptr;
        other.owned_data_ = nullptr;
        other.deleter_ = [](void*){};

        return *this;
    }

    /// Is this NDArray a view into external data?
    bool is_view() const {
        return owned_data_ == nullptr;
    }

    /// Get the value inside this `NDArray` at the given index
    ///
    /// ```
    /// auto array = NDArray(...);
    ///
    /// double value = array(2, 3, 1);
    /// ```
    template<typename ...Args>
    T operator()(Args... args) const & {
        auto index = std::array<size_t, sizeof...(Args)>{static_cast<size_t>(args)...};
        if constexpr (sizeof...(Args) != 0) {
            if (index.size() != shape_.size()) {
                throw Error(
                    "expected " + std::to_string(shape_.size()) +
                    " indexes in NDArray::operator(), got " + std::to_string(index.size())
                );
            }
        }
        return data_[details::linear_index(shape_, index)];
    }

    /// Get a reference to the value inside this `NDArray` at the given index
    ///
    /// ```
    /// auto array = NDArray(...);
    ///
    /// array(2, 3, 1) = 5.2;
    /// ```
    template<typename ...Args>
    T& operator()(Args... args) & {
        if (is_const_) {
            throw Error("This NDArray is const, can not get non const access to it");
        }

        auto index = std::array<size_t, sizeof...(Args)>{static_cast<size_t>(args)...};
        if constexpr (sizeof...(Args) != 0) {
            if (index.size() != shape_.size()) {
                throw Error(
                    "expected " + std::to_string(shape_.size()) +
                    " indexes in NDArray::operator(), got " + std::to_string(index.size())
                );
            }
        }
        return data_[details::linear_index(shape_, index)];
    }

    template<typename ...Args>
    T& operator()(Args... args) && = delete;

    /// Get the data pointer for this array, i.e. the pointer to the first
    /// element.
    const T* data() const & {
        return data_;
    }

    /// Get the data pointer for this array, i.e. the pointer to the first
    /// element.
    T* data() & {
        if (is_const_) {
            throw Error("This NDArray is const, can not get non const access to it");
        }
        return data_;
    }

    const T* data() && = delete;

    /// Get the shape of this array
    const std::vector<size_t>& shape() const & {
        return shape_;
    }

    const std::vector<size_t>& shape() && = delete;

    /// Check if this array is empty, i.e. if at least one of the shape element
    /// is 0.
    bool is_empty() const {
        for (auto s: shape_) {
            if (s == 0) {
                return true;
            }
        }
        return false;
    }

private:
    /// Create an `NDArray` from a pointer to the (row-major) data & shape.
    ///
    /// The `is_const` parameter controls whether this class should allow
    /// non-const access to the data.
    NDArray(const T* data, std::vector<size_t> shape, bool is_const):
        data_(const_cast<T*>(data)),
        shape_(std::move(shape)),
        is_const_(is_const),
        deleter_([](void*){})
    {
        validate();
    }

    /// Create a 2D NDArray from a vector of initializer lists. All the inner
    /// lists must have the same `size`.
    ///
    /// This allows creating an array (and in particular a set of Labels) with
    /// `NDArray({{1, 2}, {3, 4}, {5, 6}}, 2)`
    NDArray(const std::vector<std::initializer_list<T>>& data, size_t size):
        data_(nullptr),
        shape_({data.size(), size}),
        is_const_(false),
        deleter_([](void*){})
    {
        using vector_t = std::vector<T>;
        auto vector = std::vector<T>();
        vector.reserve(data.size() * size);
        for (const auto& row : data) {
            if (row.size() != size) {
                throw Error(
                    "invalid size for row: expected " + std::to_string(size) +
                    " got " + std::to_string(row.size())
                );
            }

            for (auto entry: row) {
                vector.emplace_back(entry);
            }
        }

        data_ = vector.data();
        owned_data_ = reinterpret_cast<void*>(new vector_t(std::move(vector)));
        deleter_ = [](void* data){
            auto data_vector = reinterpret_cast<vector_t*>(data);
            delete data_vector;
        };
        validate();
    }

    friend class Labels;

    void validate() const {
        static_assert(
            std::is_arithmetic_v<T>,
            "NDArray only works with integers and floating points"
        );

        if (shape_.empty()) {
            throw Error("invalid parameters to NDArray, shape should contain at least one element");
        }

        size_t size = 1;
        for (auto s: shape_) {
            size *= s;
        }

        if (size != 0 && data_ == nullptr) {
            throw Error("invalid parameters to NDArray, got null data pointer and non zero size");
        }
    }

    /// Pointer to the data used by this array
    T* data_ = nullptr;
    /// Full shape of this array
    std::vector<size_t> shape_ = {0, 0};
    /// Is this array const? This will dynamically prevent calling non-const
    /// function on it.
    bool is_const_ = true;
    /// Type-erased owned data for this array. This is a `nullptr` if this array
    /// is a view.
    void* owned_data_ = nullptr;
    /// Custom delete function for the `owned_data_`.
    std::function<void(void*)> deleter_;
};


/// Compare this `NDArray` with another `NDarray`. The array are equal if
/// and only if both the shape and data are equal.
template<typename T>
bool operator==(const NDArray<T>& lhs, const NDArray<T>& rhs) {
    if (lhs.shape() != rhs.shape()) {
        return false;
    }

    if constexpr (std::is_floating_point_v<T>) {
        // IEEE compatible equality for float, double, etc.
        // Element-wise, safer way with correct NaN and +/-0.0 management.
        const auto size = details::product(lhs.shape());
        return std::equal(lhs.data(), lhs.data() + size, rhs.data());
    } else {
        // A fast, bitwise memory comparison is safe and efficient here.
        return std::memcmp(lhs.data(), rhs.data(), sizeof(T) * details::product(lhs.shape())) == 0;
    }
}

/// Compare this `NDArray` with another `NDarray`. The array are equal if
/// and only if both the shape and data are equal.
template<typename T>
bool operator!=(const NDArray<T>& lhs, const NDArray<T>& rhs) {
    return !(lhs == rhs);
}
} // namespace metatensor
