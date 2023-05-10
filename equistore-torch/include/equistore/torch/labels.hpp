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

class LabelsEntryHolder;
using TorchLabelsEntry = torch::intrusive_ptr<LabelsEntryHolder>;

namespace details {
    /// Transform a torch::IValue containing either a single string, a list of
    /// strings or a tuple of strings into a `std::vector<std::string>`.
    /// `argument_name` is used in the error message.
    std::vector<std::string> normalize_names(torch::IValue names, std::string argument_name);
}

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
    /// Construct `LabelsHolder` from a set of names and the corresponding
    /// values
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

    /// Get a view of `labels` corresponding to only the given columns names
    static TorchLabels view(const TorchLabels& labels, std::vector<std::string> names);

    /// Create Labels with a single entry, and a single dimension named `"_"`
    static TorchLabels single();

    /// Create Labels with the given dimension names and zero entries
    static TorchLabels empty(torch::IValue names);

    /// Create Labels with a single dimension with the given name and values in
    /// the [0, stop) range
    static TorchLabels range(std::string name, int64_t end);

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
    ///    - a `LabelsEntry`
    ///    - a 1-D torch::Tensor containing integers;
    ///    - a list of integers;
    ///    - a tuple of integers;
    torch::optional<int64_t> position(torch::IValue entry) const;

    /// Print the names and values of these Labels to a string, including at
    /// most `max_entries` entries (set this to -1 to print all entries), and
    /// indenting all lines after the first with `indent` spaces.
    std::string print(int64_t max_entries, int64_t indent) const;

    /// Implementation of __str__ for Python
    std::string __str__() const;

    /// Implementation of __repr__ for Python
    std::string __repr__() const;

    /// Get the underlying equistore::Labels
    const equistore::Labels& as_equistore() const;

    /// Is this a LabelsView or a Labels?
    bool is_view() const {
        return !labels_.has_value();
    }

    /// Transform a view of Labels into owned Labels, which can be further given
    /// to equistore functions. This does nothing if the Labels are already
    /// owned.

    // A view is created by the `view` function (also `__getitem__` in Python),
    // and does not have a corresponding `equistore::Labels` (`labels_` is
    // `nullopt`)
    LabelsHolder to_owned() const;

private:
    /// marker type to differentiate the private constructor below from the main
    /// one
    struct CreateView {};
    /// Create a view for an existing LabelsHolder
    LabelsHolder(std::vector<std::string> names, torch::Tensor values, CreateView);

    friend class torch::intrusive_ptr<LabelsHolder>;

    /// names of the Labels, stored here for easier retrieval from Python
    std::vector<std::string> names_;

    /// Keep the values of the Labels inside a Tensor as well
    torch::Tensor values_;

    /// Underlying equistore labels, this is undefined when the Labels is
    /// actually a view (with selected columns) into another Labels
    torch::optional<equistore::Labels> labels_;
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


/// A single entry inside a TorchLabels
class EQUISTORE_TORCH_EXPORT LabelsEntryHolder: public torch::CustomClassHolder {
public:
    LabelsEntryHolder(TorchLabels labels, int64_t index):
        labels_(std::move(labels))
    {
        values_ = labels_->values()[index];
    }

    LabelsEntryHolder(const LabelsEntryHolder&) = default;
    LabelsEntryHolder& operator=(const LabelsEntryHolder&) = default;
    LabelsEntryHolder(LabelsEntryHolder&&) = default;
    LabelsEntryHolder& operator=(LabelsEntryHolder&&) = default;
    ~LabelsEntryHolder() override = default;

    /// Get the names of the dimensions/columns of these Labels
    const std::vector<std::string>& names() const {
        return labels_->names();
    }

    /// Get the values of these labels as a torch Tensor
    torch::Tensor values() const {
        return values_;
    }

    /// Get the number of dimensions in this LabelsEntry.
    ///
    /// This is the same as `values().size(0)`
    int64_t size() const {
        return values_.size(0);
    }

    /// Get the value at `index` in this `LabelsEntry`
    int32_t operator[](int64_t index) const {
        return values_[index].item<int32_t>();
    }

    /// Get the value for the `name` dimension in this `LabelsEntry`
    int32_t operator[](const std::string& name) const;

    /// implementation of __getitem__, forwarding to one of the operator[]
    int64_t getitem(torch::IValue index) const;

    /// Implementation of __repr__ for Python
    std::string __repr__() const;

private:
    torch::Tensor values_;

    TorchLabels labels_;
};

inline bool operator==(const LabelsEntryHolder& lhs, const LabelsEntryHolder& rhs) {
    std::cout << "names = " << (lhs.names() == rhs.names()) << ", values = " << torch::all(lhs.values() == rhs.values()).item<bool>() << std::endl;
    return lhs.names() == rhs.names() && torch::all(lhs.values() == rhs.values()).item<bool>();
}

inline bool operator!=(const LabelsEntryHolder& lhs, const LabelsEntryHolder& rhs) {
    return !(lhs == rhs);
}

}

#endif
