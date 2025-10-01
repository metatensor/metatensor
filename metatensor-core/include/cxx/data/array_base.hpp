#pragma once
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
                return MTS_SUCCESS;
            }, array, origin);
        };

        array.copy = [](const void* array, mts_array_t* new_array) {
            return details::catch_exceptions([](const void* array, mts_array_t* new_array){
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                auto copy = cxx_array->copy();
                *new_array = DataArrayBase::to_mts_array_t(std::move(copy));
                return MTS_SUCCESS;
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
                return MTS_SUCCESS;
            }, array, shape, shape_count, new_array);
        };

        array.data = [](void* array, double** data) {
            return details::catch_exceptions([](void* array, double** data){
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                *data = cxx_array->data();
                return MTS_SUCCESS;
            }, array, data);
        };

        array.shape = [](const void* array, const uintptr_t** shape, uintptr_t* shape_count) {
            return details::catch_exceptions([](const void* array, const uintptr_t** shape, uintptr_t* shape_count){
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                const auto& cxx_shape = cxx_array->shape();
                *shape = cxx_shape.data();
                *shape_count = static_cast<uintptr_t>(cxx_shape.size());
                return MTS_SUCCESS;
            }, array, shape, shape_count);
        };

        array.reshape = [](void* array, const uintptr_t* shape, uintptr_t shape_count) {
            return details::catch_exceptions([](void* array, const uintptr_t* shape, uintptr_t shape_count){
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                auto cxx_shape = std::vector<uintptr_t>(shape, shape + shape_count);
                cxx_array->reshape(std::move(cxx_shape));
                return MTS_SUCCESS;
            }, array, shape, shape_count);
        };

        array.swap_axes = [](void* array, uintptr_t axis_1, uintptr_t axis_2) {
            return details::catch_exceptions([](void* array, uintptr_t axis_1, uintptr_t axis_2){
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                cxx_array->swap_axes(axis_1, axis_2);
                return MTS_SUCCESS;
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
                return MTS_SUCCESS;
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
