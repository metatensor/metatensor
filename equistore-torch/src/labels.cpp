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

static torch::IValue normalize_names(torch::IValue names) {
    if (names.isString()) {
        return c10::ivalue::Tuple::create(names.toString());
    } else if (names.isList()) {
        const auto& names_list = names.toListRef();
        for (const auto& name: names_list) {
            if (!name.isString()) {
                C10_THROW_ERROR(ValueError,
                    "names must be a list of strings"
                );
            }
        }
        return c10::ivalue::Tuple::create(names_list);
    } else if (names.isTuple()) {
        for (const auto& name: names.toTupleRef().elements()) {
            if (!name.isString()) {
                C10_THROW_ERROR(ValueError,
                    "names must be a tuple of strings"
                );
            }
        }
        return names;
    } else {
        C10_THROW_ERROR(ValueError,
            "names must be a list of strings"
        );
    }
}

eqs_labels_t labels_from_torch(const torch::IValue& names, const torch::Tensor& values) {
    // extract the names from the Python IValue
    auto c_names = std::vector<const char*>();
    assert(names.isTuple());
    for (const auto& name: names.toTupleRef().elements()) {
        assert(name.isString());
        c_names.push_back(name.toString()->string().c_str());
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

torch::IValue LabelsHolder::position(torch::IValue entry) const {
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
        return torch::IValue();
    } else {
        return torch::IValue(position);
    }
}
