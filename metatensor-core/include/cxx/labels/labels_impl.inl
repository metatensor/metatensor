/// Two Labels compare equal only if they have the same names and values in the
/// same order.
inline bool operator==(const Labels& lhs, const Labels& rhs) {
    if (lhs.names_.size() != rhs.names_.size()) {
        return false;
    }

    for (size_t i=0; i<lhs.names_.size(); i++) {
        if (std::strcmp(lhs.names_[i], rhs.names_[i]) != 0) {
            return false;
        }
    }

    return lhs.values() == rhs.values();
}

/// Two Labels compare equal only if they have the same names and values in the
/// same order.
inline bool operator!=(const Labels& lhs, const Labels& rhs) {
    return !(lhs == rhs);
}
