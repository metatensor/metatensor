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
    /// Construct `Labels` from a set of names and the corresponding values
    ///
    /// The names should be either a single string or a list/tuple of strings;
    /// and the values should be a 2D tensor of integers.
    LabelsHolder(torch::IValue names, torch::Tensor values);

    /// Convenience constructor for building `Labels` in C++, similar to
    /// `equistore::Labels`.
    static TorchLabels create(
        const std::vector<std::string>& names,
        std::vector<std::initializer_list<int32_t>> values
    );

    LabelsHolder(const LabelsHolder&) = default;
    LabelsHolder& operator=(const LabelsHolder&) = default;
    LabelsHolder(LabelsHolder&&) = default;
    LabelsHolder& operator=(LabelsHolder&&) = default;
    ~LabelsHolder() override = default;

    /// Get the names of the dimensions/columns of these Labels
    ///
    /// @return a torch::IValue containing the names as a `Tuple[str]`
    torch::IValue names() const {
        return names_;
    }

    /// Get the values of these labels as a torch Tensor
    torch::Tensor values() const {
        return values_;
    }

    /// Get the number of entries in this set of Labels.
    ///
    /// This is the same as `values().sizes()[0]`
    int64_t count() const {
        return static_cast<int64_t>(labels_.count());
    }

    /// Get the number of dimensions in this set of Labels.
    ///
    /// This is the same as `values().sizes()[1]`
    int64_t size() const {
        return static_cast<int64_t>(labels_.size());
    }

    /// Get the position of the given `entry` in this set of Labels, or None if
    /// the entry is not part of these labels.
    ///
    /// @param entry one of:
    ///    - a 1-D torch::Tensor containing integers;
    ///    - a list of integers;
    ///    - a tuple of integers;
    ///
    /// @return either a 64-bit integer or `None`
    torch::IValue position(torch::IValue entry) const;

    /// Get the underlying equistore::Labels
    const equistore::Labels& as_equistore() const {
        return labels_;
    }
private:
    /// Tuple[str] containing the names of the Labels, stored as `torch::IValue`
    /// for easier retrieval from Python
    torch::IValue names_;

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
