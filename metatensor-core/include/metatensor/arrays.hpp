#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <cassert>
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
        mts_array_t* array
    );

    /**
     * @brief DLTensor resource bundle
     *
     * Owns metadata
     */
    struct DLPackContextBase {
        std::vector<int64_t> shape;
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

} // namespace details


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

        array.copy = [](const void* array, mts_array_t* new_array) {
            return details::catch_exceptions([](const void* array, mts_array_t* new_array){
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                auto copy = cxx_array->copy();
                *new_array = DataArrayBase::to_mts_array_t(std::move(copy));
            }, array, new_array);
        };

        array.create = [](const void* array, const uintptr_t* shape, uintptr_t shape_count, mts_array_t* new_array) {
            return details::catch_exceptions([](
                const void* array,
                const uintptr_t* shape,
                uintptr_t shape_count,
                mts_array_t* new_array
            ) {
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                auto cxx_shape = std::vector<size_t>();
                for (size_t i=0; i<static_cast<size_t>(shape_count); i++) {
                    cxx_shape.push_back(static_cast<size_t>(shape[i]));
                }
                auto copy = cxx_array->create(std::move(cxx_shape));
                *new_array = DataArrayBase::to_mts_array_t(std::move(copy));
            }, array, shape, shape_count, new_array);
        };

        array.data = [](void* array, double** data) {
            return details::catch_exceptions([](void* array, double** data){
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                *data = cxx_array->data();
            }, array, data);
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

        array.move_samples_from = [](
            void* array,
            const void* input,
            const mts_sample_mapping_t* samples,
            uintptr_t samples_count,
            uintptr_t property_start,
            uintptr_t property_end
        ) {
            return details::catch_exceptions([](
                void* array,
                const void* input,
                const mts_sample_mapping_t* samples,
                uintptr_t samples_count,
                uintptr_t property_start,
                uintptr_t property_end
            ) {
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                const auto* cxx_input = static_cast<const DataArrayBase*>(input);
                auto cxx_samples = std::vector<mts_sample_mapping_t>(samples, samples + samples_count);

                cxx_array->move_samples_from(*cxx_input, cxx_samples, property_start, property_end);
            }, array, input, samples, samples_count, property_start, property_end);
        };

        return array;
    }

    /// Get "data origin" for this array in.
    ///
    /// Users of `DataArrayBase` should register a single data
    /// origin with `mts_register_data_origin`, and use it for all compatible
    /// arrays.
    virtual mts_data_origin_t origin() const = 0;

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
    /// The new array should be filled with zeros.
    virtual std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape) const = 0;

    /// Get a pointer to the underlying data storage.
    ///
    /// This function is allowed to fail if the data is not accessible in RAM,
    /// not stored as 64-bit floating point values, or not stored as a
    /// C-contiguous array.
    virtual double* data() & = 0;

    double* data() && = delete;

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
    /// The `samples` indicate where the data should be moved from `input` to
    /// the current DataArrayBase.
    ///
    /// This function should copy data from `input[samples[i].input, ..., :]` to
    /// `array[samples[i].output, ..., property_start:property_end]` for `i` up
    /// to `samples_count`. All indexes are 0-based.
    virtual void move_samples_from(
        const DataArrayBase& input,
        std::vector<mts_sample_mapping_t> samples,
        uintptr_t property_start,
        uintptr_t property_end
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

    double* data() & override {
        if constexpr(std::is_same_v<T, double>){
            return reinterpret_cast<double*>(data_->data());
        } else {
            throw Error("data() needs double, use as_dlpack instead");
        }
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

    std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape) const override {
        return std::unique_ptr<DataArrayBase>(new SimpleDataArray(std::move(shape)));
    }

    void move_samples_from(
        const DataArrayBase& input,
        std::vector<mts_sample_mapping_t> samples,
        uintptr_t property_start,
        uintptr_t property_end
    ) override {
        const auto& input_array = dynamic_cast<const SimpleDataArray<T>&>(input);
        assert(input_array.shape_.size() == this->shape_.size());

        size_t property_count = property_end - property_start;
        size_t property_dim = shape_.size() - 1;
        assert(input_array.shape_[property_dim] == property_count);

        auto input_index = std::vector<size_t>(shape_.size(), 0);
        auto output_index = std::vector<size_t>(shape_.size(), 0);

        for (const auto& sample: samples) {
            input_index[0] = sample.input;
            output_index[0] = sample.output;

            if (property_dim == 1) {
                // no components
                for (size_t property_i=0; property_i<property_count; property_i++) {
                    input_index[property_dim] = property_i;
                    output_index[property_dim] = property_i + property_start;

                    auto value = (*input_array.data_)[details::linear_index(input_array.shape_, input_index)];
                    (*this->data_)[details::linear_index(shape_, output_index)] = value;
                }
            } else {
                auto last_component_dim = shape_.size() - 2;
                for (size_t component_i=1; component_i<shape_.size() - 1; component_i++) {
                    input_index[component_i] = 0;
                }

                bool done = false;
                while (!done) {
                    for (size_t component_i=1; component_i<shape_.size() - 1; component_i++) {
                        output_index[component_i] = input_index[component_i];
                    }

                    for (size_t property_i=0; property_i<property_count; property_i++) {
                        input_index[property_dim] = property_i;
                        output_index[property_dim] = property_i + property_start;

                        auto value = (*input_array.data_)[details::linear_index(input_array.shape_, input_index)];
                        (*this->data_)[details::linear_index(shape_, output_index)] = value;
                    }

                    input_index[last_component_dim] += 1;
                    for (size_t component_i=last_component_dim; component_i>2; component_i--) {
                        if (input_index[component_i] >= shape_[component_i]) {
                            input_index[component_i] = 0;
                            input_index[component_i - 1] += 1;
                        }
                    }

                    if (input_index[1] >= shape_[1]) {
                        done = true;
                    }
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

    double* data() & override {
        throw metatensor::Error("can not call `data` for an EmptyDataArray");
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

    std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape) const override {
        return std::unique_ptr<DataArrayBase>(new EmptyDataArray(std::move(shape)));
    }

    void move_samples_from(const DataArrayBase&, std::vector<mts_sample_mapping_t>, uintptr_t, uintptr_t) override {
        throw metatensor::Error("can not call `move_samples_from` for an EmptyDataArray");
    }

private:
    std::vector<uintptr_t> shape_;
};

/// Default callback for data array creating in `TensorMap::load`, which
/// will create a `SimpleDataArray<double>`.
inline mts_status_t details::default_create_array(
    const uintptr_t* shape_ptr,
    uintptr_t shape_count,
    mts_array_t* array
) {
    return details::catch_exceptions([](const uintptr_t* shape_ptr, uintptr_t shape_count, mts_array_t* array){
        auto shape = std::vector<size_t>();
        for (size_t i=0; i<shape_count; i++) {
            shape.push_back(static_cast<size_t>(shape_ptr[i]));
        }

        auto cxx_array = std::unique_ptr<DataArrayBase>(new SimpleDataArray<double>(shape));
        *array = DataArrayBase::to_mts_array_t(std::move(cxx_array));

        return MTS_SUCCESS;
    }, shape_ptr, shape_count, array);
}

} // namespace metatensor
