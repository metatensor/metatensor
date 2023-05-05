#ifndef EQUISTORE_TORCH_LABELS_HPP
#define EQUISTORE_TORCH_LABELS_HPP

#include <string>
#include <vector>

#include <torch/script.h>

#include <equistore.hpp>
#include "equistore/torch/exports.h"

namespace equistore_torch {

class LabelsHolder;
using TorchLabels = torch::intrusive_ptr<LabelsHolder>;

/// Wrapper around `equistore::Labels` for integration with TorchScript
///
/// Python/TorchScript code will typically manipulate
/// `torch::intrusive_ptr<LabelsHolder>` (i.e. `TorchLabels`) instead of
/// instances of `LabelsHolder`.
///
/// The main difference with `equistore::Labels` is that the values of the
/// labels entries are stored twice: once inside the Rust side labels, and once
/// in a `torch::Tensor`. The data inside the tensor can be moved to different
/// devices if needed.
class EQUISTORE_TORCH_EXPORT LabelsHolder: public torch::CustomClassHolder {
public:
    /// Construct `LabelsHolder` from a set of names and the corresponding values
    ///
    /// The names should be either a single string or a list/tuple of strings;
    /// and the values should be a 2D tensor of integers.
    LabelsHolder(torch::IValue names, torch::Tensor values);

    /// Convenience constructor for building `LabelsHolder` in C++, similar to
    /// `equistore::Labels`.
    static TorchLabels create(
        const std::vector<std::string>& names,
        std::vector<std::initializer_list<int32_t>> values
    );

    /// Create a `LabelsHolder` from a pre-existing `equistore::Labels`
    LabelsHolder(equistore::Labels labels);

    LabelsHolder(const LabelsHolder&) = default;
    LabelsHolder& operator=(const LabelsHolder&) = default;
    LabelsHolder(LabelsHolder&&) = default;
    LabelsHolder& operator=(LabelsHolder&&) = default;
    ~LabelsHolder() override = default;

    /// Get the names of the dimensions/columns of these Labels
    const std::vector<std::string>& names() const {
        return names_;
    }

    /// Get the values of these labels as a torch Tensor
    torch::Tensor values() const {
        return values_;
    }

    /// Get the number of entries in this set of Labels.
    ///
    /// This is the same as `values().size(0)`
    int64_t count() const {
        return values_.size(0);
    }

    /// Get the number of dimensions in this set of Labels.
    ///
    /// This is the same as `values().size(1)`
    int64_t size() const {
        return values_.size(1);
    }

    /// Get the position of the given `entry` in this set of Labels, or None if
    /// the entry is not part of these labels.
    ///
    /// @param entry one of:
    ///    - a 1-D torch::Tensor containing integers;
    ///    - a list of integers;
    ///    - a tuple of integers;
    torch::optional<int64_t> position(torch::IValue entry) const;

    /// Get the underlying equistore::Labels
    const equistore::Labels& as_equistore() const {
        return labels_;
    }
private:
    /// names of the Labels, stored here for easier retrieval from Python
    std::vector<std::string> names_;

    /// Keep the values of the Labels inside a Tensor as well
    torch::Tensor values_;

    /// Underlying equistore labels
    equistore::Labels labels_;
};

inline bool operator==(const LabelsHolder& lhs, const LabelsHolder& rhs) {
    return lhs.as_equistore() == rhs.as_equistore();
}

inline bool operator!=(const LabelsHolder& lhs, const LabelsHolder& rhs) {
    return !(lhs == rhs);
}

inline bool operator==(const LabelsHolder& lhs, const equistore::Labels& rhs) {
    return lhs.as_equistore() == rhs;
}

inline bool operator!=(const LabelsHolder& lhs, const equistore::Labels& rhs) {
    return !(lhs == rhs);
}

inline bool operator==(const equistore::Labels& lhs, const LabelsHolder& rhs) {
    return lhs == rhs.as_equistore();
}

inline bool operator!=(const equistore::Labels& lhs, const LabelsHolder& rhs) {
    return !(lhs == rhs);
}

}

#endif
