inline metatensor::Labels labels_from_cxx(
    const std::vector<std::string>& names,
    const int32_t* values,
    size_t count,
    bool assume_unique = false
) {
    mts_labels_t labels;
    std::memset(&labels, 0, sizeof(labels));

    auto c_names = std::vector<const char*>();
    for (const auto& name: names) {
        c_names.push_back(name.c_str());
    }

    labels.names = c_names.data();
    labels.size = c_names.size();
    labels.count = count;
    labels.values = values;

    if (assume_unique) {
        details::check_status(mts_labels_create_assume_unique(&labels));
    } else {
        details::check_status(mts_labels_create(&labels));
    }

    return metatensor::Labels(labels);
}
