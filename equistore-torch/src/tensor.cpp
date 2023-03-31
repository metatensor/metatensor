#include <equistore.hpp>

#include "equistore/torch/tensor.hpp"
#include "equistore/torch/array.hpp"
#include "equistore/torch/block.hpp"

using namespace equistore_torch;

TensorMapHolder::TensorMapHolder(equistore::TensorMap tensor): tensor_(std::move(tensor)) {}

static equistore::TensorBlock block_from_torch(const TorchTensorBlock& block) {
    auto components = std::vector<equistore::Labels>();
    for (const auto& component: block->components()) {
        components.push_back(component->as_equistore());
    }

    // use copy constructors of everything here, incrementing reference count
    // of the data and metadata
    auto result = equistore::TensorBlock(
        std::make_unique<TorchDataArray>(block->values()),
        block->samples()->as_equistore(),
        components,
        block->properties()->as_equistore()
    );

    for (const auto& parameter: block->gradients_list()) {
        auto gradient = block_from_torch(block->gradient(parameter));
        result.add_gradient(parameter, std::move(gradient));
    }

    return result;
}

static std::vector<equistore::TensorBlock> blocks_from_torch(const std::vector<TorchTensorBlock>& blocks) {
    auto results = std::vector<equistore::TensorBlock>();
    for (const auto& block: blocks) {
        results.emplace_back(block_from_torch(block));
    }
    return results;
}


TensorMapHolder::TensorMapHolder(TorchLabels keys, const std::vector<TorchTensorBlock>& blocks):
    tensor_(keys->as_equistore(), blocks_from_torch(std::move(blocks)))
{}

TorchTensorMap TensorMapHolder::copy() const {
    return torch::make_intrusive<TensorMapHolder>(this->tensor_.clone());
}

TorchLabels TensorMapHolder::keys() const {
    return torch::make_intrusive<LabelsHolder>(this->tensor_.keys());
}

std::vector<int64_t> TensorMapHolder::blocks_matching(const TorchLabels& selection) const {
    auto results = tensor_.blocks_matching(selection->as_equistore());

    auto results_int64 = std::vector<int64_t>();
    results_int64.reserve(results.size());
    for (auto matching: results) {
        results_int64.push_back(static_cast<int64_t>(matching));
    }

    return results_int64;
}

TorchTensorBlock TensorMapHolder::block_by_id(int64_t index) {
    return torch::make_intrusive<TensorBlockHolder>(tensor_.block_by_id(index));
}

/// Transform a torch::IValue containing either a single string, a list of
/// strings or a tuple of strings to something C++ can use
///
/// The `context` is used for the error message in case the torch::IValue is
/// none of the above.
static std::vector<std::string> extract_list_str(const torch::IValue& keys_to_move, std::string context) {
    if (keys_to_move.isString()) {
        return {keys_to_move.toString()->string()};
    } else if (keys_to_move.isList()) {
        auto results = std::vector<std::string>();
        for (const auto& element: keys_to_move.toListRef()) {
            if (element.isString()) {
                results.push_back(element.toString()->string());
            } else {
                C10_THROW_ERROR(TypeError, context + " must be a list of `str`");
            }
        }
        return results;
    } else if (keys_to_move.isTuple()) {
        auto results = std::vector<std::string>();
        for (const auto& element: keys_to_move.toTupleRef().elements()) {
            if (element.isString()) {
                results.push_back(element.toString()->string());
            } else {
                C10_THROW_ERROR(TypeError, context + " must be a tuple of `str`");
            }
        }
        return results;
    } else {
        throw std::runtime_error(
            "internal error: called extract_list_str, but IValue is not a list of str"
        );
    }
}

TorchTensorMap TensorMapHolder::keys_to_properties(torch::IValue keys_to_move, bool sort_samples) const {
    if (keys_to_move.isString() || keys_to_move.isList() || keys_to_move.isTuple()) {
        auto selection = extract_list_str(keys_to_move, "TensorMap::keys_to_properties first argument");
        auto tensor = tensor_.keys_to_properties(selection, sort_samples);
        return torch::make_intrusive<TensorMapHolder>(std::move(tensor));
    } else if (keys_to_move.isCustomClass()) {
        auto selection = keys_to_move.toCustomClass<LabelsHolder>();
        auto tensor = tensor_.keys_to_properties(selection->as_equistore(), sort_samples);
        return torch::make_intrusive<TensorMapHolder>(std::move(tensor));
    } else {
        C10_THROW_ERROR(TypeError,
            "TensorMap::keys_to_properties first argument must be a `str`, list of `str` or `Labels`"
        );
    }
}

TorchTensorMap TensorMapHolder::keys_to_samples(torch::IValue keys_to_move, bool sort_samples) const {
    if (keys_to_move.isString() || keys_to_move.isList() || keys_to_move.isTuple()) {
        auto selection = extract_list_str(keys_to_move, "TensorMap::keys_to_samples first argument");
        auto tensor = tensor_.keys_to_samples(selection, sort_samples);
        return torch::make_intrusive<TensorMapHolder>(std::move(tensor));
    } else if (keys_to_move.isCustomClass()) {
        auto selection = keys_to_move.toCustomClass<LabelsHolder>();
        auto tensor = tensor_.keys_to_samples(selection->as_equistore(), sort_samples);
        return torch::make_intrusive<TensorMapHolder>(std::move(tensor));
    } else {
        C10_THROW_ERROR(TypeError,
            "TensorMap::keys_to_samples first argument must be a `str`, list of `str` or `Labels`"
        );
    }
}

TorchTensorMap TensorMapHolder::components_to_properties(torch::IValue dimensions) const {
    auto selection = extract_list_str(dimensions, "TensorMap::components_to_properties argument");
    auto tensor = this->tensor_.components_to_properties(selection);
    return torch::make_intrusive<TensorMapHolder>(std::move(tensor));
}

torch::IValue TensorMapHolder::sample_names() {
    if (this->keys()->as_equistore().count() == 0) {
        return c10::ivalue::Tuple::create({});
    }

    return this->block_by_id(0)->samples()->names();
}

torch::IValue TensorMapHolder::components_names() {
    auto result = c10::List<torch::IValue>(c10::TupleType::create({c10::getTypePtr<c10::IValue>()}));

    if (this->keys()->as_equistore().count() != 0) {
        for (const auto& component: this->block_by_id(0)->components()) {
            result.push_back(component->names());
        }
    }

    return result;
}

torch::IValue TensorMapHolder::property_names() {
    if (this->keys()->as_equistore().count() == 0) {
        return c10::ivalue::Tuple::create({});
    }

    return this->block_by_id(0)->properties()->names();
}
