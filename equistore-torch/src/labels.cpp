#include <cassert>

#include <torch/torch.h>

#include <equistore.hpp>

#include "equistore/torch/labels.hpp"

using namespace equistore_torch;

/// Check that `values` is a `shape_length`-dimensional contiguous array of
/// 32-bit integers on CPU, or convert it to be so.
static torch::Tensor normalize_int32_tensor(torch::Tensor values, size_t shape_length, const std::string& context) {
    if (!torch::can_cast(values.scalar_type(), torch::kI32)) {
        C10_THROW_ERROR(ValueError,
            context + " must be an Tensor of 32-bit integers"
        );
    }

    if (values.sizes().size() != shape_length) {
        C10_THROW_ERROR(ValueError,
            context + " must be a " + std::to_string(shape_length) + "D Tensor"
        );
    }

    return values.to(torch::kCPU, torch::kI32).contiguous();
}

static torch::Tensor initializer_list_to_tensor(
    std::vector<std::initializer_list<int32_t>> values,
    size_t size
) {
    auto vector = std::vector<int32_t>();

    auto count = values.size();
    vector.reserve(count * size);
    for (auto row: std::move(values)) {
        if (row.size() != size) {
            C10_THROW_ERROR(ValueError,
                "invalid size for row: expected " + std::to_string(size) +
                " got " + std::to_string(row.size())
            );
        }

        for (auto entry: row) {
            vector.emplace_back(entry);
        }
    }

    auto tensor = torch::tensor(std::move(vector));
    return tensor.reshape({static_cast<int64_t>(count), static_cast<int64_t>(size)});
}

static std::vector<std::string> normalize_names(torch::IValue names) {
    auto results = std::vector<std::string>();
    if (names.isString()) {
        results.push_back(names.toStringRef());
    } else if (names.isList()) {
        const auto& names_list = names.toListRef();
        for (const auto& name: names_list) {
            if (!name.isString()) {
                C10_THROW_ERROR(ValueError,
                    "names must be a list of strings"
                );
            }
            results.push_back(name.toStringRef());
        }
    } else if (names.isTuple()) {
        for (const auto& name: names.toTupleRef().elements()) {
            if (!name.isString()) {
                C10_THROW_ERROR(ValueError,
                    "names must be a tuple of strings"
                );
            }
            results.push_back(name.toStringRef());
        }
    } else {
        C10_THROW_ERROR(ValueError,
            "names must be a list of strings"
        );
    }
    return results;
}

static eqs_labels_t labels_from_torch(const std::vector<std::string>& names, const torch::Tensor& values) {
    // extract the names from the Python IValue
    auto c_names = std::vector<const char*>();
    for (const auto& name: names) {
        c_names.push_back(name.c_str());
    }

    // check the values
    assert(values.sizes().size() == 2);
    assert(values.scalar_type() == torch::kI32);
    assert(values.is_contiguous());
    assert(values.device().is_cpu());
    if (values.sizes()[1] != c_names.size()) {
        C10_THROW_ERROR(ValueError,
            "invalid Labels: the names must have an entry for each column of the array"
        );
    }

    // create the C labels
    eqs_labels_t labels;
    std::memset(&labels, 0, sizeof(labels));

    labels.names = c_names.data();
    labels.size = c_names.size();
    labels.count = values.sizes()[0];
    labels.values = static_cast<const int32_t*>(values.data_ptr());

    equistore::details::check_status(eqs_labels_create(&labels));

    return labels;
}

static std::vector<std::string> names_from_equistore(const equistore::Labels& labels) {
    auto names = std::vector<std::string>();
    for (const auto name: labels.names()) {
        names.push_back(std::string(name));
    }
    return names;
}


static torch::Tensor values_from_equistore(const equistore::Labels& labels) {
    auto sizes = std::vector<int64_t>();
    for (auto dim: labels.shape()) {
        sizes.push_back(static_cast<int64_t>(dim));
    }

    return torch::from_blob(
        // Unfortunately, we can not prevent writing to this tensor since torch
        // does not support read-only tensor:
        // https://github.com/pytorch/pytorch/issues/44027
        //
        // TODO: should we make a copy here instead?
        const_cast<int32_t*>(labels.data()),
        sizes,
        torch::TensorOptions().dtype(torch::kInt32)
    );
}

LabelsHolder::LabelsHolder(torch::IValue names, torch::Tensor values):
    names_(normalize_names(names)),
    values_(normalize_int32_tensor(std::move(values), 2, "Labels values")),
    labels_(labels_from_torch(names_, values_))
{}


TorchLabels LabelsHolder::create(
    const std::vector<std::string>& names,
    std::vector<std::initializer_list<int32_t>> values
) {
    return torch::make_intrusive<LabelsHolder>(
        torch::IValue(names),
        initializer_list_to_tensor(values, names.size())
    );
}

LabelsHolder::LabelsHolder(equistore::Labels labels):
    names_(names_from_equistore(labels)),
    values_(values_from_equistore(labels)),
    labels_(std::move(labels))
{}


torch::optional<int64_t> LabelsHolder::position(torch::IValue entry) const {
    int64_t position = -1;
    if (entry.isTensor()) {
        auto tensor = normalize_int32_tensor(entry.toTensor(), 1, "entry passed to Labels::position");
        position = labels_.position(
            static_cast<const int32_t*>(tensor.data_ptr()),
            tensor.sizes()[0]
        );
    } else if (entry.isIntList()) {
        auto int32_values = std::vector<int32_t>();
        for (const auto& value: entry.toIntList()) {
            int32_values.push_back(static_cast<int32_t>(value));
        }
        position = labels_.position(int32_values);
    } else if (entry.isList()) {
        auto int32_values = std::vector<int32_t>();
        for (auto value: entry.toListRef()) {
            if (value.isInt()) {
                int32_values.push_back(static_cast<int32_t>(value.toInt()));
            } else {
                C10_THROW_ERROR(TypeError,
                    "parameter to Labels::positions must be a tensor or list/tuple of integers"
                );
            }
        }
        position = labels_.position(int32_values);
    } else if (entry.isTuple()) {
        auto int32_values = std::vector<int32_t>();
        for (auto value: entry.toTupleRef().elements()) {
            if (value.isInt()) {
                int32_values.push_back(static_cast<int32_t>(value.toInt()));
            } else {
                C10_THROW_ERROR(TypeError,
                    "parameter to Labels::positions must be a tensor or list/tuple of integers"
                );
            }
        }
        position = labels_.position(int32_values);
    } else {
        C10_THROW_ERROR(TypeError,
            "parameter to Labels::positions must be a tensor or list/tuple of integers"
        );
    }

    if (position == -1) {
        return {};
    } else {
        return position;
    }
}
