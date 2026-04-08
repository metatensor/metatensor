#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include <metatensor.h>

#include "./errors.hpp"

namespace metatensor {
namespace details {
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

    mts_status_t default_create_array(
        const uintptr_t* shape_ptr,
        uintptr_t shape_count,
        DLDataType dtype,
        mts_array_t* array
    );

    /**
     * @brief DLTensor resource bundle
     *
     * Owns metadata
     */
    struct DLPackContextBase {
        /// Shape of the array
        std::vector<int64_t> shape;
        /// Strides of the array, in number of elements (not bytes)
        std::vector<int64_t> strides;
        virtual ~DLPackContextBase() = default;
    };

    /**
     * @brief Derived and typed DLTensor storage context
     *
     * Owns a pointer to a shared buffer
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    struct DLPackContext : DLPackContextBase {
        /// Shared pointer to the data buffer, which keeps the data alive as
        /// long as required.
        std::shared_ptr<std::vector<T>> data;
    };

    /**
     * @brief C style deleter for DLPack
     *
     * Destroys metadata and decrements the shared pointer.
     */
    DLPACK_EXTERN_C inline void DLPackDeleter(DLManagedTensorVersioned* self){
        if (self != nullptr) {
            // manager_ctx points to a DLPackContextBase (allocated with new)
            auto* ctx = static_cast<DLPackContextBase*>(self->manager_ctx);
            // avoid reuse after free
            self->manager_ctx = nullptr;
            // delete is polymorphic so it will destroys the derived DLPackContext<T>
            delete ctx;
            delete self;
        }
    }

    /// Map a C++ arithmetic type to the corresponding DLDataType.
    template<typename T>
    DLDataType dtype_of() {
        static_assert(std::is_arithmetic_v<T>, "dtype_of requires an arithmetic type");
        DLDataType dt;
        dt.lanes = 1;
        dt.bits = static_cast<uint8_t>(sizeof(T) * 8);
        if constexpr (std::is_floating_point_v<T>) {
            dt.code = kDLFloat;
        } else if constexpr (std::is_signed_v<T>) {
            dt.code = kDLInt;
        } else {
            dt.code = kDLUInt;
        }
        return dt;
    }

    /// Check whether a DLDataType matches the C++ type T.
    ///
    /// Returns true when code, bits **and** lanes all match.
    template<typename T>
    bool dlpack_dtype_matches(DLDataType dtype) {
        if (dtype.lanes != 1) {
            return false;
        }
        if constexpr (std::is_floating_point_v<T>) {
            return dtype.code == kDLFloat
                && dtype.bits == static_cast<uint8_t>(sizeof(T) * 8);
        } else if constexpr (std::is_integral_v<T>) {
            if constexpr (std::is_signed_v<T>) {
                return dtype.code == kDLInt
                    && dtype.bits == static_cast<uint8_t>(sizeof(T) * 8);
            } else {
                return dtype.code == kDLUInt
                    && dtype.bits == static_cast<uint8_t>(sizeof(T) * 8);
            }
        }
        return false;
    }

} // namespace details


/// Read-only N-dimensional array that owns a DLPack managed tensor.
///
/// `DLPackArray<T>` takes ownership of a `DLManagedTensorVersioned*` and calls
/// its deleter on destruction. It exposes a read-only view of the underlying
/// data (`.data()`, `.shape()`, `operator()`).
///
/// Move-only: copy construction and copy assignment are deleted.
template<typename T>
class DLPackArray {
public:
    /// Create an empty `DLPackArray`, with shape `[0, 0]`.
    DLPackArray(): managed_(nullptr), data_(nullptr), shape_({0, 0}) {}

    /// Create a `DLPackArray` that takes ownership of `managed`.
    ///
    /// Validates that the DLPack dtype matches `T` (code, bits, and
    /// `lanes == 1`). Throws `metatensor::Error` on mismatch and cleans up
    /// the managed tensor before throwing.
    explicit DLPackArray(DLManagedTensorVersioned* managed):
        managed_(managed),
        data_(nullptr)
    {
        if (managed_ == nullptr) {
            shape_ = {0, 0};
            return;
        }

        auto& tensor = managed_->dl_tensor;

        if (!details::dlpack_dtype_matches<T>(tensor.dtype)) {
            // Copy dtype before calling deleter, which frees managed_
            auto dtype = tensor.dtype;
            if (managed_->deleter != nullptr) {
                managed_->deleter(managed_);
            }
            managed_ = nullptr;
            throw Error(
                "DLPackArray dtype mismatch: DLPack tensor has dtype "
                "(code=" + std::to_string(dtype.code)
                + ", bits=" + std::to_string(dtype.bits)
                + ", lanes=" + std::to_string(dtype.lanes)
                + ") which does not match the requested C++ type (sizeof="
                + std::to_string(sizeof(T)) + ")"
            );
        }

        data_ = reinterpret_cast<T*>(
            static_cast<char*>(tensor.data) + tensor.byte_offset
        );
        for (int32_t i = 0; i < tensor.ndim; ++i) {
            shape_.push_back(static_cast<size_t>(tensor.shape[i]));
        }
    }

    ~DLPackArray() {
        if (managed_ != nullptr && managed_->deleter) {
            managed_->deleter(managed_);
        }
    }

    /// DLPackArray is not copy-constructible
    DLPackArray(const DLPackArray&) = delete;
    /// DLPackArray can not be copy-assigned
    DLPackArray& operator=(const DLPackArray&) = delete;

    /// DLPackArray is move-constructible
    DLPackArray(DLPackArray&& other) noexcept: DLPackArray() {
        *this = std::move(other);
    }

    /// DLPackArray can be move-assigned
    DLPackArray& operator=(DLPackArray&& other) noexcept {
        if (managed_ != nullptr && managed_->deleter) {
            managed_->deleter(managed_);
        }

        managed_ = other.managed_;
        data_ = other.data_;
        shape_ = std::move(other.shape_);

        other.managed_ = nullptr;
        other.data_ = nullptr;
        other.shape_ = {0, 0};

        return *this;
    }

    /// Get the DLDevice for this array.
    ///
    /// Returns `{kDLCPU, 0}` if the array is empty (null managed tensor).
    DLDevice device() const {
        if (managed_ == nullptr) {
            return {kDLCPU, 0};
        }
        return managed_->dl_tensor.device;
    }

    /// Get the value inside this `DLPackArray` at the given index
    ///
    /// ```
    /// auto array = DLPackArray<double>(...);
    ///
    /// double value = array(2, 3, 1);
    /// ```
    template<typename ...Args>
    T operator()(Args... args) const & {
        if (managed_ != nullptr && managed_->dl_tensor.device.device_type != kDLCPU) {
            throw Error(
                "can not index into a DLPackArray on a non-CPU device"
            );
        }
        auto index = std::array<size_t, sizeof... (Args)>{static_cast<size_t>(args)...};
        if (index.size() != shape_.size()) {
            throw Error(
                "expected " + std::to_string(shape_.size()) +
                " indexes in DLPackArray::operator(), got " + std::to_string(index.size())
            );
        }
        return data_[details::linear_index(shape_, index)];
    }

    template<typename ...Args>
    T operator()(Args... args) && = delete;

    /// Get the data pointer for this array, i.e. the pointer to the first
    /// element.
    const T* data() const & {
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
    /// Owned DLPack managed tensor (may be nullptr)
    DLManagedTensorVersioned* managed_;
    /// Pointer to the data inside the managed tensor
    T* data_;
    /// Shape extracted from the DLTensor
    std::vector<size_t> shape_;
};


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

    ~NDArray() {
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
        auto index = std::array<size_t, sizeof... (Args)>{static_cast<size_t>(args)...};
        if (index.size() != shape_.size()) {
            throw Error(
                "expected " + std::to_string(shape_.size()) +
                " indexes in NDArray::operator(), got " + std::to_string(index.size())
            );
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

        auto index = std::array<size_t, sizeof... (Args)>{static_cast<size_t>(args)...};
        if (index.size() != shape_.size()) {
            throw Error(
                "expected " + std::to_string(shape_.size()) +
                " indexes in NDArray::operator(), got " + std::to_string(index.size())
            );
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
        for (auto row: std::move(data)) {
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

    if (std::is_integral_v<T>) {
        // compare integers with memcmp
        return std::memcmp(lhs.data(), rhs.data(), sizeof(T) * details::product(lhs.shape())) == 0;
    } else {
        // make sure to handle NaN and ±0.0 correctly when comparing floating
        // point data and everything else
        const auto* lhs_ptr = lhs.data();
        const auto* rhs_ptr = rhs.data();
        for (size_t i=0; i<details::product(lhs.shape()); i++) {
            if (lhs_ptr[i] != rhs_ptr[i]) {
                return false;
            }
        }
        return true;
    }
}

/// Compare this `NDArray` with another `NDarray`. The array are equal if
/// and only if both the shape and data are equal.
template<typename T>
bool operator!=(const NDArray<T>& lhs, const NDArray<T>& rhs) {
    return !(lhs == rhs);
}

/// `DataArrayBase` manages n-dimensional arrays used as data in a block or
/// tensor map. The array itself if opaque to this library and can come from
/// multiple sources: Rust program, a C/C++ program, a Fortran program, Python
/// with numpy or torch. The data does not have to live on CPU, or even on the
/// same machine where this code is executed.
///
/// **WARNING**: all function implementations **MUST** be thread-safe, and can
/// be called from multiple threads at the same time. The `DataArrayBase` itself
/// might be moved from one thread to another.
class DataArrayBase {
public:
    DataArrayBase() = default;
    virtual ~DataArrayBase() = default;

    /// DataArrayBase can be copy-constructed
    DataArrayBase(const DataArrayBase&) = default;
    /// DataArrayBase can be copy-assigned
    DataArrayBase& operator=(const DataArrayBase&) = default;
    /// DataArrayBase can be move-constructed
    DataArrayBase(DataArrayBase&&) noexcept = default;
    /// DataArrayBase can be move-assigned
    DataArrayBase& operator=(DataArrayBase&&) noexcept = default;

    /// Convert a concrete `DataArrayBase` to a C-compatible `mts_array_t`
    ///
    /// The `mts_array_t` takes ownership of the data, which should be released
    /// with `mts_array_t::destroy`.
    static mts_array_t to_mts_array_t(std::unique_ptr<DataArrayBase> data) {
        mts_array_t array;
        std::memset(&array, 0, sizeof(array));

        array.ptr = data.release();

        array.destroy = [](void* array) {
            auto ptr = std::unique_ptr<DataArrayBase>(static_cast<DataArrayBase*>(array));
            // let ptr go out of scope
        };

        array.origin = [](const void* array, mts_data_origin_t* origin) {
            return details::catch_exceptions([](const void* array, mts_data_origin_t* origin){
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                *origin = cxx_array->origin();
            }, array, origin);
        };

        array.device = [](const void* array, DLDevice* device) {
            return details::catch_exceptions([](const void* array, DLDevice* device){
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                *device = cxx_array->device();
            }, array, device);
        };

        array.dtype = [](const void* array, DLDataType* dtype) {
            return details::catch_exceptions([](const void* array, DLDataType* dtype){
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                *dtype = cxx_array->dtype();
            }, array, dtype);
        };

        array.copy = [](const void* array, mts_array_t* new_array) {
            return details::catch_exceptions([](const void* array, mts_array_t* new_array){
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                auto copy = cxx_array->copy();
                *new_array = DataArrayBase::to_mts_array_t(std::move(copy));
            }, array, new_array);
        };

        array.create = [](const void* array, const uintptr_t* shape, uintptr_t shape_count, mts_array_t fill_value, mts_array_t* new_array) {
            return details::catch_exceptions([](
                const void* array,
                const uintptr_t* shape,
                uintptr_t shape_count,
                mts_array_t fill_value,
                mts_array_t* new_array
            ) {
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                auto cxx_shape = std::vector<size_t>();
                for (size_t i=0; i<static_cast<size_t>(shape_count); i++) {
                    cxx_shape.push_back(static_cast<size_t>(shape[i]));
                }
                auto copy = cxx_array->create(std::move(cxx_shape), fill_value);
                *new_array = DataArrayBase::to_mts_array_t(std::move(copy));
            }, array, shape, shape_count, fill_value, new_array);
        };

        array.as_dlpack = [](
            void *array,
            DLManagedTensorVersioned **dl_managed_tensor,
            DLDevice device,
            const int64_t *stream,
            DLPackVersion max_version
        ) {
            return details::catch_exceptions(
                [](
                    void *array,
                    DLManagedTensorVersioned **dl_managed_tensor,
                    DLDevice device,
                    const int64_t *stream,
                    DLPackVersion max_version
                ) {
                    auto *cxx_arr = static_cast<DataArrayBase *>(array);
                    *dl_managed_tensor = cxx_arr->as_dlpack(device, stream, max_version);
                },
                array, dl_managed_tensor, device, stream, max_version);
        };

        array.shape = [](const void* array, const uintptr_t** shape, uintptr_t* shape_count) {
            return details::catch_exceptions([](const void* array, const uintptr_t** shape, uintptr_t* shape_count){
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                const auto& cxx_shape = cxx_array->shape();
                *shape = cxx_shape.data();
                *shape_count = static_cast<uintptr_t>(cxx_shape.size());
            }, array, shape, shape_count);
        };

        array.reshape = [](void* array, const uintptr_t* shape, uintptr_t shape_count) {
            return details::catch_exceptions([](void* array, const uintptr_t* shape, uintptr_t shape_count){
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                auto cxx_shape = std::vector<uintptr_t>(shape, shape + shape_count);
                cxx_array->reshape(std::move(cxx_shape));
            }, array, shape, shape_count);
        };

        array.swap_axes = [](void* array, uintptr_t axis_1, uintptr_t axis_2) {
            return details::catch_exceptions([](void* array, uintptr_t axis_1, uintptr_t axis_2){
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                cxx_array->swap_axes(axis_1, axis_2);
            }, array, axis_1, axis_2);
        };

        array.move_data = [](
            void* array,
            const void* input,
            const mts_data_movement_t* moves,
            uintptr_t moves_count
        ) {
            return details::catch_exceptions([](
                void* array,
                const void* input,
                const mts_data_movement_t* moves,
                uintptr_t moves_count
            ) {
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                const auto* cxx_input = static_cast<const DataArrayBase*>(input);
                auto cxx_moves = std::vector<mts_data_movement_t>(moves, moves + moves_count);

                cxx_array->move_data(*cxx_input, std::move(cxx_moves));
            }, array, input, moves, moves_count);
        };

        return array;
    }

    /// Get "data origin" for this array in.
    ///
    /// Users of `DataArrayBase` should register a single data
    /// origin with `mts_register_data_origin`, and use it for all compatible
    /// arrays.
    virtual mts_data_origin_t origin() const = 0;

    /// Get the device where this array's data resides.
    virtual DLDevice device() const = 0;

    /// Get the data type of this array.
    ///
    /// This provides the `dtype` vtable callback for metatensor-core.
    /// If not overridden, the dtype falls back to extracting it from
    /// `as_dlpack`, which is more expensive. Implementations should
    /// override this for better performance.
    virtual DLDataType dtype() const = 0;

    /// Get a DLPack representation of this array
    ///
    /// The returned pointer is owned by the caller and should be freed
    /// using its deleter function when no longer needed.
    ///
    /// See the documentation of `mts_array_t::as_dlpack` for more details about
    /// the parameters.
    virtual DLManagedTensorVersioned* as_dlpack(
        DLDevice device,
        const int64_t* stream,
        DLPackVersion max_version
    ) = 0;

    /// Make a copy of this DataArrayBase and return the new array. The new
    /// array is expected to have the same data origin and parameters (data
    /// type, data location, etc.)
    virtual std::unique_ptr<DataArrayBase> copy() const = 0;

    /// Create a new array with the same options as the current one (data type,
    /// data location, etc.) and the requested `shape`.
    ///
    /// The new array should be filled with the scalar value from `fill_value`,
    /// which must be an `mts_array_t` with shape `(1,)` and the same dtype as
    /// this array. This function should call `fill_value.destroy` if the
    /// function pointer is not null when `fill_value` is no longer needed.
    virtual std::unique_ptr<DataArrayBase> create(
        std::vector<uintptr_t> shape,
        mts_array_t fill_value
    ) const = 0;

    /// Get the shape of this array
    virtual const std::vector<uintptr_t>& shape() const & = 0;

    const std::vector<uintptr_t>& shape() && = delete;

    /// Set the shape of this array to the given `shape`
    virtual void reshape(std::vector<uintptr_t> shape) = 0;

    /// Swap the axes `axis_1` and `axis_2` in this `array`.
    virtual void swap_axes(uintptr_t axis_1, uintptr_t axis_2) = 0;

    /// Set entries in the current array taking data from the `input` array.
    ///
    /// This array is guaranteed to be created by calling `mts_array_t::create`
    /// with one of the arrays in the same block or tensor map as the `input`.
    ///
    /// The `moves` indicate where the data should be moved from `input` to the
    /// current DataArrayBase.
    ///
    /// This function should copy data from `input[move.sample_in, ...,
    /// move.properties_start_in + i]` to `array[move.sample_out, ...,
    /// move.properties_start_out + i]` for each `move` in `moves` and `i` up to
    /// `move.properties_length`. All indexes are 0-based.
    virtual void move_data(
        const DataArrayBase& input,
        std::vector<mts_data_movement_t> moves
    ) = 0;
};

/// Very basic implementation of DataArrayBase in C++.
///
/// This is included as an example implementation of DataArrayBase, and to make
/// metatensor usable without additional dependencies. For other uses cases, it
/// might be better to implement DataArrayBase on your data, using
/// functionalities from `Eigen`, `Boost.Array`, etc.
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class SimpleDataArray : public metatensor::DataArrayBase {
public:
    /// Create a SimpleDataArray with the given `shape`, and all elements set to
    /// `value`
        SimpleDataArray(std::vector<uintptr_t> shape, T value = T{}):
        shape_(std::move(shape)), data_(std::make_shared<std::vector<T>>(details::product(shape_), value)) {}

    /// Create a SimpleDataArray with the given `shape` and `data`.
    ///
    /// The data is interpreted as a row-major n-dimensional array.
    SimpleDataArray(std::vector<uintptr_t> shape, std::vector<T> data):
        shape_(std::move(shape)),
        data_(std::make_shared<std::vector<T>>(std::move(data)))
    {
        if (data_->size() != details::product(shape_)) {
            throw Error("the shape and size of the data don't match in SimpleDataArray");
        }
    }

    ~SimpleDataArray() override = default;

    /// SimpleDataArray can be copy-constructed
    SimpleDataArray(const SimpleDataArray&) = default;
    /// SimpleDataArray can be copy-assigned
    SimpleDataArray& operator=(const SimpleDataArray&) = default;
    /// SimpleDataArray can be move-constructed
    SimpleDataArray(SimpleDataArray&&) noexcept = default;
    /// SimpleDataArray can be move-assigned
    SimpleDataArray& operator=(SimpleDataArray&&) noexcept = default;

    mts_data_origin_t origin() const override {
        mts_data_origin_t origin = 0;
        mts_register_data_origin("metatensor::SimpleDataArray", &origin);
        return origin;
    }

    DLDevice device() const override {
        return {kDLCPU, 0};
    }

    DLDataType dtype() const override {
        return details::dtype_of<T>();
    }

    const std::vector<uintptr_t>& shape() const & override {
        return shape_;
    }

    void reshape(std::vector<uintptr_t> shape) override {
        if (details::product(shape_) != details::product(shape)) {
            throw metatensor::Error("invalid shape in reshape");
        }
        shape_ = std::move(shape);
    }

    void swap_axes(uintptr_t axis_1, uintptr_t axis_2) override {
        auto new_data = std::vector<T>(details::product(shape_), T{});
        auto new_shape = shape_;
        std::swap(new_shape[axis_1], new_shape[axis_2]);

        for (size_t i=0; i<details::product(shape_); i++) {
            auto index = details::cartesian_index(shape_, i);
            std::swap(index[axis_1], index[axis_2]);

            new_data[details::linear_index(new_shape, index)] = (*data_)[i];
        }

        shape_ = std::move(new_shape);
        data_ = std::make_shared<std::vector<T>>(std::move(new_data));
    }

    std::unique_ptr<DataArrayBase> copy() const override {
        return std::unique_ptr<DataArrayBase>(new SimpleDataArray(*this));
    }

    std::unique_ptr<DataArrayBase> create(
        std::vector<uintptr_t> shape,
        mts_array_t fill_value
    ) const override {
        DLDevice cpu_device = {kDLCPU, 0};
        DLPackVersion max_version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        DLManagedTensorVersioned* fill_value_dlpack = nullptr;
        auto status = fill_value.as_dlpack(fill_value.ptr, &fill_value_dlpack, cpu_device, nullptr, max_version);
        if (status != MTS_SUCCESS) {
            throw Error("failed to extract fill_value as DLPack");
        }

        // Validate fill_value shape from the DLPack tensor directly
        if (fill_value_dlpack->dl_tensor.ndim != 0) {
            if (fill_value_dlpack->deleter != nullptr) {
                fill_value_dlpack->deleter(fill_value_dlpack);
            }
            throw Error("`fill_value` must be a single scalar");
        }

        T scalar;
        auto code = fill_value_dlpack->dl_tensor.dtype.code;
        auto bits = fill_value_dlpack->dl_tensor.dtype.bits;

        // Account for DLPack byte_offset per spec
        const auto* data = static_cast<const char*>(fill_value_dlpack->dl_tensor.data);
        data += fill_value_dlpack->dl_tensor.byte_offset;

        if (code == kDLFloat && bits == 64) {
            scalar = static_cast<T>(*reinterpret_cast<const double*>(data));
        } else if (code == kDLFloat && bits == 32) {
            scalar = static_cast<T>(*reinterpret_cast<const float*>(data));
        } else if (code == kDLFloat && bits == 16) {
            // f16: read as uint16_t and convert (no native C++ f16 type)
            auto raw = *reinterpret_cast<const uint16_t*>(data);
            // IEEE 754 f16->f32 conversion
            uint32_t sign = (raw >> 15) & 0x1;
            uint32_t exp = (raw >> 10) & 0x1F;
            uint32_t frac = raw & 0x3FF;
            float val;
            if (exp == 0) {
                // zero and subnormals: value = (-1)^sign * 2^(-14) * (frac/1024)
                val = std::ldexp(static_cast<float>(frac), -24);
                if (sign != 0) {
                    val = -val;
                }
            } else if (exp == 0x1F) {
                uint32_t f32 = (sign << 31) | 0x7F800000 | (frac << 13);
                std::memcpy(&val, &f32, sizeof(float));
            } else {
                uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
                std::memcpy(&val, &f32, sizeof(float));
            }
            scalar = static_cast<T>(val);
        } else if (code == kDLInt && bits == 64) {
            scalar = static_cast<T>(*reinterpret_cast<const int64_t*>(data));
        } else if (code == kDLInt && bits == 32) {
            scalar = static_cast<T>(*reinterpret_cast<const int32_t*>(data));
        } else if (code == kDLInt && bits == 16) {
            scalar = static_cast<T>(*reinterpret_cast<const int16_t*>(data));
        } else if (code == kDLInt && bits == 8) {
            scalar = static_cast<T>(*reinterpret_cast<const int8_t*>(data));
        } else if (code == kDLUInt && bits == 64) {
            scalar = static_cast<T>(*reinterpret_cast<const uint64_t*>(data));
        } else if (code == kDLUInt && bits == 32) {
            scalar = static_cast<T>(*reinterpret_cast<const uint32_t*>(data));
        } else if (code == kDLUInt && bits == 16) {
            scalar = static_cast<T>(*reinterpret_cast<const uint16_t*>(data));
        } else if (code == kDLUInt && bits == 8) {
            scalar = static_cast<T>(*reinterpret_cast<const uint8_t*>(data));
        } else if (code == kDLBool && bits == 8) {
            scalar = static_cast<T>(*reinterpret_cast<const bool*>(data));
        } else {
            if (fill_value_dlpack->deleter != nullptr) {
                fill_value_dlpack->deleter(fill_value_dlpack);
            }
            throw Error("unsupported fill_value dtype");
        }

        if (fill_value_dlpack->deleter != nullptr) {
            fill_value_dlpack->deleter(fill_value_dlpack);
        }

        if (fill_value.destroy != nullptr) {
            fill_value.destroy(fill_value.ptr);
        }

        return std::unique_ptr<DataArrayBase>(new SimpleDataArray(std::move(shape), scalar));
    }

    void move_data(
        const DataArrayBase& input,
        std::vector<mts_data_movement_t> moves
    ) override {
        const auto& input_array = dynamic_cast<const SimpleDataArray<T>&>(input);
        assert(input_array.shape_.size() == this->shape_.size());

        if (moves.empty()) {
            return;
        }

        size_t property_dim = shape_.size() - 1;

        // Calculate the total number of components (product of dimensions 1 to property_dim - 1)
        size_t num_components = 1;
        for (size_t i = 1; i < property_dim; i++) {
            num_components *= shape_[i];
        }

        // The distance between two consecutive component blocks in memory
        // This is equal to the size of the property dimension
        size_t input_component_stride = input_array.shape_[property_dim];
        size_t output_component_stride = shape_[property_dim];

        // Calculate the stride for the samples (dimension 0)
        // This is the number of elements in one sample, including all components and properties
        size_t input_sample_stride = input_component_stride * num_components;
        size_t output_sample_stride = output_component_stride * num_components;

        for (const auto& move: moves) {
            size_t input_sample_offset = move.sample_in * input_sample_stride;
            size_t output_sample_offset = move.sample_out * output_sample_stride;

            if (move.properties_length == input_component_stride && move.properties_length == output_component_stride) {
                // Optimization: if we are moving the full set of properties, we can
                // move all components at once since they are contiguous in memory.
                std::copy_n(
                    input_array.data_->begin() + static_cast<std::ptrdiff_t>(input_sample_offset + move.properties_start_in),
                    input_sample_stride,
                    this->data_->begin() + static_cast<std::ptrdiff_t>(output_sample_offset + move.properties_start_out)
                );
            } else {
                for (size_t i = 0; i < num_components; i++) {
                    size_t input_offset = input_sample_offset + (i * input_component_stride) + move.properties_start_in;
                    size_t output_offset = output_sample_offset + (i * output_component_stride) + move.properties_start_out;

                    std::copy_n(
                        input_array.data_->begin() + static_cast<std::ptrdiff_t>(input_offset),
                        move.properties_length,
                        this->data_->begin() + static_cast<std::ptrdiff_t>(output_offset)
                    );
                }
            }
        }
    }

    /// Get a const view of the data managed by this SimpleDataArray
    NDArray<T> view() const {
        return NDArray<T>(data_->data(), shape_);
    }

    /// Get a mutable view of the data managed by this SimpleDataArray
    NDArray<T> view() {
        return NDArray<T>(data_->data(), shape_);
    }

    /// Extract a reference to SimpleDataArray out of an `mts_array_t`.
    ///
    /// This function fails if the `mts_array_t` does not contain a
    /// SimpleDataArray.
    static SimpleDataArray& from_mts_array(mts_array_t& array) {
        mts_data_origin_t origin = 0;
        auto status = array.origin(array.ptr, &origin);
        if (status != MTS_SUCCESS) {
            throw Error("failed to get data origin");
        }

        std::array<char, 64> buffer = {0};
        status = mts_get_data_origin(origin, buffer.data(), buffer.size());
        if (status != MTS_SUCCESS || std::string(buffer.data()) != "metatensor::SimpleDataArray") {
            throw Error("this array is not a metatensor::SimpleDataArray");
        }

        auto* base = static_cast<DataArrayBase*>(array.ptr);
        return dynamic_cast<SimpleDataArray&>(*base);
    }

    /// Extract a const reference to SimpleDataArray out of an `mts_array_t`.
    ///
    /// This function fails if the `mts_array_t` does not contain a
    /// SimpleDataArray.
    static const SimpleDataArray& from_mts_array(const mts_array_t& array) {
        mts_data_origin_t origin = 0;
        auto status = array.origin(array.ptr, &origin);
        if (status != MTS_SUCCESS) {
            throw Error("failed to get data origin");
        }

        std::array<char, 64> buffer = {0};
        status = mts_get_data_origin(origin, buffer.data(), buffer.size());
        if (status != MTS_SUCCESS || std::string(buffer.data()) != "metatensor::SimpleDataArray") {
            throw Error("this array is not a metatensor::SimpleDataArray");
        }

        const auto* base = static_cast<const DataArrayBase*>(array.ptr);
        return dynamic_cast<const SimpleDataArray&>(*base);
    }

    DLManagedTensorVersioned *as_dlpack(
        DLDevice device,
        const int64_t* stream,
        DLPackVersion max_version
    ) override {
        if (device.device_type != kDLCPU) {
            throw Error("SimpleDataArray only supports CPU device (kDLCPU)");
        }

        if (stream != nullptr) {
            throw Error("`stream` must be null for CPU data");
        }

        DLPackVersion mta_version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        // SEMVER, so major must match, and the MTA minor must be below the minor(caller)
        bool major_mismatch = max_version.major != mta_version.major;
        bool minor_too_high = max_version.minor < mta_version.minor;
        if (major_mismatch || minor_too_high) {
            throw Error(
                "SimpleDataArray supports DLPack version " +
                std::to_string(mta_version.major) + "." +
                std::to_string(mta_version.minor) +
                ". Caller requested incompatible version " +
                std::to_string(max_version.major) + "." +
                std::to_string(max_version.minor)
            );
        }

        using metatensor::details::DLPackContext;
        using metatensor::details::DLPackDeleter;
        auto ctx = std::make_unique<DLPackContext<T>>();
        // fill shape
        ctx->shape.resize(this->shape_.size());
        std::transform(this->shape_.begin(), this->shape_.end(), ctx->shape.begin(),
                       [](uintptr_t s) { return static_cast<int64_t>(s); });
        // fill strides (C contiguous, element strides)
        ctx->strides.resize(this->shape_.size());
        if (!this->shape_.empty()) {
            ctx->strides.back() = 1;
            for (size_t i = this->shape_.size() - 1; i > 0; --i) {
                ctx->strides[i - 1] = ctx->strides[i] * ctx->shape[i];
            }
        }
        // point to existing data and setup manager
        ctx->data = this->data_;
        auto managed = std::make_unique<DLManagedTensorVersioned>();
        managed->version = mta_version;
        managed->flags = 0;
        managed->deleter = DLPackDeleter;

        // Fill the DLTensor view
        auto &tensor = managed->dl_tensor;
        tensor.device = {kDLCPU, 0};
        tensor.ndim = static_cast<int32_t>(this->shape_.size());

        if (std::is_floating_point_v<T>) {
            tensor.dtype.code = kDLFloat;
            tensor.dtype.bits = static_cast<uint8_t>(sizeof(T) * 8);
            tensor.dtype.lanes = 1;
        } else if (std::is_integral_v<T>) {
            if (std::is_signed_v<T>) {
                tensor.dtype.code = kDLInt;
            } else {
                tensor.dtype.code = kDLUInt;
            }
            tensor.dtype.bits = static_cast<uint8_t>(sizeof(T) * 8);
            tensor.dtype.lanes = 1;
        } else {
            static_assert(std::is_arithmetic_v<T>, "unsupported tensor element type");
        }

        tensor.byte_offset = 0;

        // tensor.data now points into ctx->data's underlying vector.
        tensor.data = const_cast<void*>(static_cast<const void*>(ctx->data->data()));
        if (tensor.ndim == 0) {
            tensor.shape = nullptr;
            tensor.strides = nullptr;
        } else {
            tensor.shape = ctx->shape.data();
            tensor.strides = ctx->strides.empty() ? nullptr : ctx->strides.data();
        }

        // Transfer Ownership to C pointers (release unique_ptr)
        managed->manager_ctx = static_cast<void*>(ctx.release());
        return managed.release();
    }

private:
    std::vector<uintptr_t> shape_;
    std::shared_ptr<std::vector<T>> data_;

    template <typename U>
    friend bool operator==(const SimpleDataArray<U>& lhs, const SimpleDataArray<U>& rhs);
};

/// Two SimpleDataArray compare as equal if they have the exact same shape and
/// data.
template<typename U>
inline bool operator==(const SimpleDataArray<U>& lhs, const SimpleDataArray<U>& rhs) {
    return lhs.shape_ == rhs.shape_ && *lhs.data_ == *rhs.data_;
}

/// Two SimpleDataArray compare as equal if they have the exact same shape and
/// data.
template<typename U>
inline bool operator!=(const SimpleDataArray<U>& lhs, const SimpleDataArray<U>& rhs) {
    return !(lhs == rhs);
}

/// An implementation of `DataArrayBase` containing no data.
///
/// This class only tracks it's shape, and can be used when only the metadata
/// of a `TensorBlock` is important, leaving the data unspecified.
class EmptyDataArray: public metatensor::DataArrayBase {
public:
    /// Create ae `EmptyDataArray` with the given `shape`
    EmptyDataArray(std::vector<uintptr_t> shape):
        shape_(std::move(shape)) {}

    ~EmptyDataArray() override = default;

    /// EmptyDataArray can be copy-constructed
    EmptyDataArray(const EmptyDataArray&) = default;
    /// EmptyDataArray can be copy-assigned
    EmptyDataArray& operator=(const EmptyDataArray&) = default;
    /// EmptyDataArray can be move-constructed
    EmptyDataArray(EmptyDataArray&&) noexcept = default;
    /// EmptyDataArray can be move-assigned
    EmptyDataArray& operator=(EmptyDataArray&&) noexcept = default;

    mts_data_origin_t origin() const override {
        mts_data_origin_t origin = 0;
        mts_register_data_origin("metatensor::EmptyDataArray", &origin);
        return origin;
    }

    DLDevice device() const override {
        return {kDLCPU, 0};
    }

    DLDataType dtype() const override {
        // EmptyDataArray defaults to f64 for consistency with default_create_array
        return details::dtype_of<double>();
    }

    DLManagedTensorVersioned *as_dlpack(DLDevice, const int64_t*, DLPackVersion) override {
        throw metatensor::Error("can not call `as_dlpack` for an EmtpyDataArray");
    }

    const std::vector<uintptr_t>& shape() const & override {
        return shape_;
    }

    void reshape(std::vector<uintptr_t> shape) override {
        if (details::product(shape_) != details::product(shape)) {
            throw metatensor::Error("invalid shape in reshape");
        }
        shape_ = std::move(shape);
    }

    void swap_axes(uintptr_t axis_1, uintptr_t axis_2) override {
        std::swap(shape_[axis_1], shape_[axis_2]);
    }

    std::unique_ptr<DataArrayBase> copy() const override {
        return std::unique_ptr<DataArrayBase>(new EmptyDataArray(*this));
    }

    std::unique_ptr<DataArrayBase> create(
        std::vector<uintptr_t> shape,
        mts_array_t /*fill_value*/
    ) const override {
        return std::unique_ptr<DataArrayBase>(new EmptyDataArray(std::move(shape)));
    }

    void move_data(const DataArrayBase&, std::vector<mts_data_movement_t>) override {
        throw metatensor::Error("can not call `move_data` for an EmptyDataArray");
    }

private:
    std::vector<uintptr_t> shape_;
};

/// Default callback for data array creating in `TensorMap::load`.
/// Dispatches on the `dtype` parameter to create the appropriate
/// `SimpleDataArray<T>`.
inline mts_status_t details::default_create_array(
    const uintptr_t* shape_ptr,
    uintptr_t shape_count,
    DLDataType dtype,
    mts_array_t* array
) {
    return details::catch_exceptions([](const uintptr_t* shape_ptr, uintptr_t shape_count, DLDataType dtype, mts_array_t* array){
        auto shape = std::vector<size_t>();
        for (size_t i=0; i<shape_count; i++) {
            shape.push_back(static_cast<size_t>(shape_ptr[i]));
        }

        if (dtype.lanes != 1) {
            throw metatensor::Error(
                "unsupported DLDataType in default_create_array: lanes=" +
                std::to_string(dtype.lanes) + " (expected 1)"
            );
        }

        std::unique_ptr<DataArrayBase> cxx_array;
        if (dtype.code == kDLFloat && dtype.bits == 64) {
            cxx_array.reset(new SimpleDataArray<double>(shape));
        } else if (dtype.code == kDLFloat && dtype.bits == 32) {
            cxx_array.reset(new SimpleDataArray<float>(shape));
        } else if (dtype.code == kDLInt && dtype.bits == 8) {
            cxx_array.reset(new SimpleDataArray<int8_t>(shape));
        } else if (dtype.code == kDLInt && dtype.bits == 16) {
            cxx_array.reset(new SimpleDataArray<int16_t>(shape));
        } else if (dtype.code == kDLInt && dtype.bits == 32) {
            cxx_array.reset(new SimpleDataArray<int32_t>(shape));
        } else if (dtype.code == kDLInt && dtype.bits == 64) {
            cxx_array.reset(new SimpleDataArray<int64_t>(shape));
        } else if (dtype.code == kDLUInt && dtype.bits == 8) {
            cxx_array.reset(new SimpleDataArray<uint8_t>(shape));
        } else if (dtype.code == kDLUInt && dtype.bits == 16) {
            cxx_array.reset(new SimpleDataArray<uint16_t>(shape));
        } else if (dtype.code == kDLUInt && dtype.bits == 32) {
            cxx_array.reset(new SimpleDataArray<uint32_t>(shape));
        } else if (dtype.code == kDLUInt && dtype.bits == 64) {
            cxx_array.reset(new SimpleDataArray<uint64_t>(shape));
        } else {
            throw metatensor::Error(
                "unsupported DLDataType in default_create_array: code="
                + std::to_string(dtype.code) + " bits=" + std::to_string(dtype.bits)
            );
        }

        *array = DataArrayBase::to_mts_array_t(std::move(cxx_array));

        return MTS_SUCCESS;
    }, shape_ptr, shape_count, dtype, array);
}

} // namespace metatensor
