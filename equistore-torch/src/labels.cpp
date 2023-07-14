#include <cassert>

#include <torch/torch.h>

#include <equistore.hpp>

#include "equistore/torch/labels.hpp"

using namespace equistore_torch;

/// Check that `values` is a `shape_length`-dimensional array of 32-bit
/// integers, or convert it to be so.
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

    return values.to(torch::kI32);
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

std::vector<std::string> equistore_torch::details::normalize_names(torch::IValue names, std::string argument_name) {
    auto results = std::vector<std::string>();
    if (names.isString()) {
        results.push_back(names.toStringRef());
    } else if (names.isList()) {
        const auto& names_list = names.toListRef();
        for (const auto& name: names_list) {
            if (!name.isString()) {
                C10_THROW_ERROR(TypeError,
                    argument_name + " must be a list of strings, got element with type '"
                    + name.type()->str() + "' instead"
                );
            }
            results.push_back(name.toStringRef());
        }
    } else if (names.isTuple()) {
        for (const auto& name: names.toTupleRef().elements()) {
            if (!name.isString()) {
                C10_THROW_ERROR(TypeError,
                    argument_name + " must be a tuple of strings, got element with type '"
                    + name.type()->str() + "' instead"
                );
            }
            results.push_back(name.toStringRef());
        }
    } else {
        C10_THROW_ERROR(TypeError,
            argument_name + " must be a string, list of strings or tuple of string, got '"
            + names.type()->str() + "' instead"
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


static torch::Tensor values_from_equistore(equistore::Labels& labels) {
    // check if the labels are already associated with a tensor
    auto user_data = labels.user_data();
    if (user_data != nullptr) {
        // if we start using user_data for more than this exact case (storing
        // tensors inside Labels), this code might fails and will need to start
        // checking that `user_data` is actually a tensor.
        return *static_cast<torch::Tensor*>(user_data);
    }

    // otherwise create a new tensor
    auto sizes = std::vector<int64_t>();
    for (auto dim: labels.shape()) {
        sizes.push_back(static_cast<int64_t>(dim));
    }

    return torch::from_blob(
        // This should really be a `const int32_t*`, but we can not prevent
        // writing to this tensor since torch does not support read-only tensor:
        // https://github.com/pytorch/pytorch/issues/44027
        const_cast<int32_t*>((const_cast<const equistore::Labels&>(labels)).data()),
        sizes,
        torch::TensorOptions().dtype(torch::kInt32)
    );
}

LabelsHolder::LabelsHolder(torch::IValue names, torch::Tensor values):
    names_(details::normalize_names(names, "names")),
    values_(normalize_int32_tensor(std::move(values), 2, "Labels values")),
    labels_(equistore::Labels(labels_from_torch(names_, values_.to(torch::kCPU).contiguous())))
{
    // register the torch tensor as a custom user data in the labels
    auto user_data = equistore::LabelsUserData(
        new torch::Tensor(values_),
        [](void* tensor) { delete static_cast<torch::Tensor*>(tensor); }
    );

    labels_->set_user_data(std::move(user_data));
}

TorchLabels LabelsHolder::create(
    const std::vector<std::string>& names,
    std::vector<std::initializer_list<int32_t>> values
) {
    return torch::make_intrusive<LabelsHolder>(
        torch::IValue(names),
        initializer_list_to_tensor(values, names.size())
    );
}


LabelsHolder::LabelsHolder(std::vector<std::string> names, torch::Tensor values, CreateView):
    names_(std::move(names)),
    values_(std::move(values))
{}

TorchLabels LabelsHolder::view(const TorchLabels& labels, std::vector<std::string> names) {
    if (names.empty()) {
        C10_THROW_ERROR(ValueError,
            "can not index Labels with an empty list of dimension names"
        );
    }

    auto dimensions = std::vector<int64_t>();
    for (const auto& name: names) {
        auto it = std::find(std::begin(labels->names_), std::end(labels->names_), name);
        if (it == std::end(labels->names_)) {
            C10_THROW_ERROR(ValueError,
                "'" + name + "' not found in the dimensions of these Labels"
            );
        }

        auto index = std::distance(std::begin(labels->names_), it);
        dimensions.push_back(static_cast<int64_t>(index));
    }

    auto new_values = labels->values_.index({torch::indexing::Slice(), torch::tensor(std::move(dimensions))});
    return torch::make_intrusive<LabelsHolder>(std::move(names), std::move(new_values), CreateView{});
}

LabelsHolder::LabelsHolder(equistore::Labels labels):
    names_(names_from_equistore(labels)),
    values_(values_from_equistore(labels)),
    labels_(std::move(labels))
{}


TorchLabels LabelsHolder::single() {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto values = torch::tensor({0}, options).reshape({1, 1});
    return torch::make_intrusive<LabelsHolder>("_", std::move(values));
}


TorchLabels LabelsHolder::empty(torch::IValue names_ivalue) {
    auto names = details::normalize_names(names_ivalue, "empty");
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto values = torch::tensor(std::vector<int>(), options).reshape({0, static_cast<int64_t>(names.size())});
    return torch::make_intrusive<LabelsHolder>(std::move(names), std::move(values));
}


TorchLabels LabelsHolder::range(std::string name, int64_t end) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto values = torch::arange(end, options).reshape({end, 1});
    return torch::make_intrusive<LabelsHolder>(std::move(name), std::move(values));
}

torch::Tensor LabelsHolder::column(std::string dimension) {
    auto it = std::find(std::begin(names_), std::end(names_), dimension);

    if (it == std::end(names_)) {
        C10_THROW_ERROR(ValueError,
            "'" + dimension + "' not found in the dimensions of these Labels"
        );
    }

    int64_t index = std::distance(std::begin(names_), it);
    return values_.index({torch::indexing::Slice(), index});
}

const equistore::Labels& LabelsHolder::as_equistore() const {
    if (!labels_.has_value()) {
        C10_THROW_ERROR(ValueError,
            "can not call this function on Labels view, call to_owned first"
        );
    }

    return labels_.value();
}

LabelsHolder LabelsHolder::to_owned() const {
    if (labels_.has_value()) {
        return *this;
    } else {
        return LabelsHolder(this->names_, values_);
    }
}

TorchLabels LabelsHolder::append(std::string name, torch::Tensor values) {
    return this->insert(this->size(), std::move(name), std::move(values));
}


TorchLabels LabelsHolder::insert(int64_t index, std::string name, torch::Tensor values) {
    auto names = this->names();

    auto it = std::begin(names) + index;
    names.insert(it, name);

    if (values.sizes().size() != 1) {
        C10_THROW_ERROR(
            ValueError,
            "`values` must be a 1D tensor"
        );
    }

    auto old_values = this->values();
    auto first = old_values.index({torch::indexing::Slice(), torch::indexing::Slice(0, index)});
    auto second = old_values.index({torch::indexing::Slice(), torch::indexing::Slice(index)});

    auto new_values = torch::hstack({first, values.reshape({values.size(0), 1}), second});

    return torch::make_intrusive<LabelsHolder>(std::move(names), std::move(new_values));
}


TorchLabels LabelsHolder::remove(std::string name) {
    auto names = this->names();

    auto it = std::find(std::begin(names), std::end(names), name);
    if (it == std::end(names)) {
        C10_THROW_ERROR(
            ValueError,
            "'" + name + "' not found in the dimensions of these Labels"
        );
    }

    auto column_index = it - std::begin(names);
    names.erase(it);

    auto values = this->values();
    auto first = values.index({torch::indexing::Slice(), torch::indexing::Slice(0, column_index)});
    auto second = values.index({torch::indexing::Slice(), torch::indexing::Slice(column_index + 1)});
    auto new_values = torch::hstack({first, second});

    return torch::make_intrusive<LabelsHolder>(std::move(names), std::move(new_values));
}


TorchLabels LabelsHolder::rename(std::string old_name, std::string new_name) {
    auto names = this->names();

    auto it = std::find(std::begin(names), std::end(names), old_name);
    if (it == std::end(names)) {
        C10_THROW_ERROR(
            ValueError,
            "'" + old_name + "' not found in the dimensions of these Labels"
        );
    }
    auto column_index = it - std::begin(names);

    names[column_index] = new_name;
    return torch::make_intrusive<LabelsHolder>(std::move(names), std::move(this->values()));
}

void LabelsHolder::to(torch::Device device) {
    // first move the values
    values_ = values_.to(device);

    // then make sure that when accessing these labels again we still have the
    // values on the same device by updating the registered user data
    auto user_data = equistore::LabelsUserData(
        new torch::Tensor(values_),
        [](void* tensor) { delete static_cast<torch::Tensor*>(tensor); }
    );

    labels_->set_user_data(std::move(user_data));
}

torch::optional<int64_t> LabelsHolder::position(torch::IValue entry) const {
    const auto& labels = this->as_equistore();

    int64_t position = -1;
    if (entry.isCustomClass()) {
        const auto& labels_entry = entry.toCustomClass<LabelsEntryHolder>();
        auto values = labels_entry->values().to(torch::kCPU).contiguous();
        position = labels.position(
            static_cast<const int32_t*>(values.data_ptr()),
            values.size(0)
        );
    } else if (entry.isTensor()) {
        auto tensor = normalize_int32_tensor(entry.toTensor(), 1, "entry passed to Labels::position");
        tensor = tensor.to(torch::kCPU).contiguous();
        position = labels.position(
            static_cast<const int32_t*>(tensor.data_ptr()),
            tensor.size(0)
        );
    } else if (entry.isIntList()) {
        auto int32_values = std::vector<int32_t>();
        for (const auto& value: entry.toIntList()) {
            int32_values.push_back(static_cast<int32_t>(value));
        }
        position = labels.position(int32_values);
    } else if (entry.isList()) {
        auto int32_values = std::vector<int32_t>();
        for (auto value: entry.toListRef()) {
            if (value.isInt()) {
                int32_values.push_back(static_cast<int32_t>(value.toInt()));
            } else {
                C10_THROW_ERROR(TypeError,
                    "list parameter to Labels::positions must be list of integers, "
                    "got element with type '" + entry.type()->str() + "'"
                );
            }
        }
        position = labels.position(int32_values);
    } else if (entry.isTuple()) {
        auto int32_values = std::vector<int32_t>();
        for (auto value: entry.toTupleRef().elements()) {
            if (value.isInt()) {
                int32_values.push_back(static_cast<int32_t>(value.toInt()));
            } else {
                C10_THROW_ERROR(TypeError,
                    "tuple parameter to Labels::positions must be a tuple of integers, "
                    "got element with type '" + entry.type()->str() + "'"
                );
            }
        }
        position = labels.position(int32_values);
    } else {
        C10_THROW_ERROR(TypeError,
            "parameter to Labels::positions must be a LabelsEntry, tensor, or list/tuple of integers, "
            "got '" + entry.type()->str() + "' instead"
        );
    }

    if (position == -1) {
        return {};
    } else {
       return position;
    }
}

TorchLabels LabelsHolder::set_union(const TorchLabels& other) const {
    if (!labels_.has_value() || !other->labels_.has_value()) {
        C10_THROW_ERROR(ValueError,
            "can not call this function on Labels view, call to_owned first"
        );
    }

    auto result = labels_->set_union(other->labels_.value());
    return torch::make_intrusive<LabelsHolder>(std::move(result));
}

std::tuple<TorchLabels, torch::Tensor, torch::Tensor> LabelsHolder::union_and_mapping(const TorchLabels& other) const {
    if (!labels_.has_value() || !other->labels_.has_value()) {
        C10_THROW_ERROR(ValueError,
            "can not call this function on Labels view, call to_owned first"
        );
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto first_mapping = torch::zeros({this->count()}, options);
    auto second_mapping = torch::zeros({other->count()}, options);

    auto result = labels_->set_union(
        other->labels_.value(),
        first_mapping.data_ptr<int64_t>(),
        first_mapping.size(0),
        second_mapping.data_ptr<int64_t>(),
        second_mapping.size(0)
    );
    auto torch_result = torch::make_intrusive<LabelsHolder>(std::move(result));


    return std::make_tuple<TorchLabels, torch::Tensor, torch::Tensor>(
        std::move(torch_result),
        std::move(first_mapping),
        std::move(second_mapping)
    );
}

TorchLabels LabelsHolder::set_intersection(const TorchLabels& other) const {
    if (!labels_.has_value() || !other->labels_.has_value()) {
        C10_THROW_ERROR(ValueError,
            "can not call this function on Labels view, call to_owned first"
        );
    }

    auto result = labels_->set_intersection(other->labels_.value());
    return torch::make_intrusive<LabelsHolder>(std::move(result));
}

std::tuple<TorchLabels, torch::Tensor, torch::Tensor> LabelsHolder::intersection_and_mapping(const TorchLabels& other) const {
    if (!labels_.has_value() || !other->labels_.has_value()) {
        C10_THROW_ERROR(ValueError,
            "can not call this function on Labels view, call to_owned first"
        );
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto first_mapping = torch::zeros({this->count()}, options);
    auto second_mapping = torch::zeros({other->count()}, options);

    auto result = labels_->set_intersection(
        other->labels_.value(),
        first_mapping.data_ptr<int64_t>(),
        first_mapping.size(0),
        second_mapping.data_ptr<int64_t>(),
        second_mapping.size(0)
    );
    auto torch_result = torch::make_intrusive<LabelsHolder>(std::move(result));


    return std::make_tuple<TorchLabels, torch::Tensor, torch::Tensor>(
        std::move(torch_result),
        std::move(first_mapping),
        std::move(second_mapping)
    );
}

struct LabelsPrintData {
    LabelsPrintData(const std::vector<std::string>& names) {
        for (const auto& name: names) {
            // use at least one space on each side of the name
            this->widths.push_back(name.size() + 2);
        }
    }

    /// widths of each column
    std::vector<size_t> widths;
    /// first half of the values
    std::vector<std::vector<std::string>> values_first;
    /// second half of the values
    std::vector<std::vector<std::string>> values_second;

    void add_values_first(torch::Tensor entry) {
        assert(entry.sizes().size() == 1);
        assert(this->widths.size() == entry.size(0));

        auto n_elements = this->widths.size();
        auto strings = std::vector<std::string>();
        strings.reserve(n_elements);

        for (int i=0; i<n_elements; i++) {
            auto entry_str = std::to_string(entry[i].item<int32_t>());
            this->widths[i] = std::max(entry_str.size() + 2, this->widths[i]);

            strings.emplace_back(std::move(entry_str));
        }

        this->values_first.emplace_back(std::move(strings));
    }

    void add_values_second(torch::Tensor entry) {
        assert(entry.sizes().size() == 1);
        assert(this->widths.size() == entry.size(0));

        auto n_elements = this->widths.size();
        auto strings = std::vector<std::string>();
        strings.reserve(n_elements);

        for (int i=0; i<n_elements; i++) {
            auto entry_str = std::to_string(entry[i].item<int32_t>());
            this->widths[i] = std::max(entry_str.size() + 2, this->widths[i]);

            strings.emplace_back(std::move(entry_str));
        }

        this->values_second.emplace_back(std::move(strings));
    }
};

void print_string_center(std::ostringstream& output, std::string string, size_t width, bool last) {
    assert(string.length() < width);

    auto delta = width - string.size();
    auto n_before = delta / 2;
    auto n_after = delta - n_before;
    string.insert(0, n_before, ' ');

    if (!last) {
        // don't add spaces after the last element
        string.insert(string.size(), n_after, ' ');
    }

    output << string;
}

std::string LabelsHolder::print(int64_t max_entries, int64_t indent) const {
    auto print_data = LabelsPrintData(names_);

    auto n_elements = this->count();
    if (max_entries < 0 || n_elements <= max_entries) {
        for (int i=0; i<n_elements; i++) {
            print_data.add_values_first(values_[i]);
        }
    } else {
        if (max_entries < 2) {
            max_entries = 2;
        }

        auto n_after = max_entries / 2;
        auto n_before = max_entries - n_after;

        for (int i=0; i<n_before; i++) {
            print_data.add_values_first(values_[i]);
        }

        for (int i=(n_elements - n_after); i<n_elements; i++) {
            print_data.add_values_second(values_[i]);
        }
    }

    auto output = std::ostringstream();
    auto indent_str = std::string(indent, ' ');

    auto n_dimensions = this->size();
    for (int i=0; i<n_dimensions; i++) {
        auto last = i == n_dimensions - 1;
        print_string_center(output, names_[i], print_data.widths[i], last);
    }
    output << '\n';

    for (auto strings: std::move(print_data.values_first)) {
        output << indent_str;
        for (int i=0; i<n_dimensions; i++) {
            auto last = i == n_dimensions - 1;
            print_string_center(output, std::move(strings[i]), print_data.widths[i], last);
        }
        output << '\n';
    }


    if (!print_data.values_second.empty()) {
        auto half_header_widths = 0;
        for (auto w: print_data.widths) {
            half_header_widths += w;
        }
        half_header_widths /= 2;

        if (half_header_widths > 3) {
            // 3 characters in '...'
            half_header_widths -= 3;
        }
        output << indent_str << std::string(half_header_widths + 1, ' ') << "...\n";

        for (auto strings: std::move(print_data.values_second)) {
            output << indent_str;
            for (int i=0; i<n_dimensions; i++) {
                auto last = i == n_dimensions - 1;
                print_string_center(output, std::move(strings[i]), print_data.widths[i], last);
            }
            output << '\n';
        }
    }

    // remove the final \n
    auto output_string = output.str();
    assert(output_string[output_string.size() - 1] == '\n');
    output_string.pop_back();

    return output_string;
}

std::string LabelsHolder::__str__() const {
    auto output = std::ostringstream();
    if (labels_.has_value()) {
        output << "Labels(\n   ";
    } else {
        output << "LabelsView(\n   ";
    }

    output << this->print(4, 3) << "\n)";
    return output.str();
}

std::string LabelsHolder::__repr__() const {
    auto output = std::ostringstream();
    if (labels_.has_value()) {
        output << "Labels(\n   ";
    } else {
        output << "LabelsView(\n   ";
    }

    output << this->print(-1, 3) << "\n)";
    return output.str();
}

/******************************************************************************/

std::string LabelsEntryHolder::print() const {
    auto output = std::stringstream();

    output << "(";
    for (size_t i=0; i<this->size(); i++) {
        output << this->names()[i] << "=" << values_[i].item<int32_t>();

        if (i < this->size() - 1) {
            output << ", ";
        }
    }
    output << ")";

    return output.str();
}

std::string LabelsEntryHolder::__repr__() const {
    return "LabelsEntry" + this->print();
}

int32_t LabelsEntryHolder::operator[](const std::string& name) const {
    const auto& names = labels_->names();
    auto it = std::find(std::begin(names), std::end(names), name);
    if (it == std::end(names)) {
        C10_THROW_ERROR(ValueError,
            "'" + name + "' not found in the dimensions of this LabelsEntry"
        );
    }

    auto index = std::distance(std::begin(names), it);
    return this->operator[](index);
}


int64_t LabelsEntryHolder::__getitem__(torch::IValue index) const {
    if (index.isInt()) {
        return static_cast<int64_t>(this->operator[](index.toInt()));
    } else if (index.isString()) {
        return static_cast<int64_t>(this->operator[](index.toStringRef()));
    } else {
        C10_THROW_ERROR(TypeError,
            "LabelsEntry can only be indexed by int or str, got '"
            + index.type()->str() + "' instead"
        );
    }
}
