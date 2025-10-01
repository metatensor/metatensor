/// Default callback for data array creating in `TensorMap::load`, which
/// will create a `SimpleDataArray`.
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

        auto cxx_array = std::unique_ptr<DataArrayBase>(new SimpleDataArray(shape));
        *array = DataArrayBase::to_mts_array_t(std::move(cxx_array));

        return MTS_SUCCESS;
    }, shape_ptr, shape_count, array);
}
