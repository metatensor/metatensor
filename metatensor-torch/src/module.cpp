#include <torch/script.h>

#include "metatensor/torch/labels.hpp"
#include "metatensor/torch/block.hpp"
#include "metatensor/torch/tensor.hpp"


#include "metatensor/torch/module.hpp"


using namespace metatensor_torch;

template <typename T>
bool is_custom_class(torch::IValue ivalue) {
    if (ivalue.isCustomClass()) {
        // this is inspired by the code inside `torch::IValue.toCustomClass<T>()`
        auto* expected_type = c10::getCustomClassType<torch::intrusive_ptr<T>>().get();
        return ivalue.type().get() == static_cast<const c10::Type*>(expected_type);
    } else {
        return false;
    }
}

static std::pair<torch::IValue, bool> ivalue_to(
    torch::IValue ivalue,
    const torch::optional<at::Device>& device,
    const torch::optional<at::ScalarType>& dtype,
    bool non_blocking
) {
    if (is_custom_class<LabelsHolder>(ivalue)) {
        if (device.has_value()) {
            auto labels = ivalue.toCustomClass<LabelsHolder>();
            labels = labels->to(device.value(), non_blocking);
            return std::make_pair(labels, true);
        } else {
            return std::make_pair(ivalue, true);
        }
    } else if (is_custom_class<TensorBlockHolder>(ivalue)) {
        auto block = ivalue.toCustomClass<TensorBlockHolder>();
        block = block->to(dtype, device, non_blocking);
        return std::make_pair(block, true);
    } else if (is_custom_class<TensorMapHolder>(ivalue)) {
        auto tensor = ivalue.toCustomClass<TensorMapHolder>();
        tensor = tensor->to(dtype, device, non_blocking);
        return std::make_pair(tensor, true);
    } else if (ivalue.isGenericDict()) {
        auto dict = ivalue.toGenericDict();
        if (dict.empty()) {
            return std::make_pair(ivalue, true);
        }

        auto updated = c10::impl::GenericDict(dict.keyType(), dict.valueType());
        auto all_changed = true;
        auto some_changed = false;
        for (const auto& item: dict) {
            auto [updated_value, changed] = ivalue_to(item.value(), device, dtype, non_blocking);
            all_changed &= changed;
            some_changed |= changed;
            updated.insert(item.key(), updated_value);
        }
        if (some_changed) {
            if (!all_changed) {
                C10_THROW_ERROR(ValueError,
                    "dict containing both metatensor and non-metatensor data "
                    "as values are not supported"
                );
            }
            return std::make_pair(updated, true);
        }
    } else if (ivalue.isList()) {
        const auto& list = ivalue.toList();
        if (list.empty()) {
            return std::make_pair(ivalue, true);
        }

        auto updated = c10::impl::GenericList(list.elementType());
        auto all_changed = true;
        auto some_changed = false;
        for (const auto& item: list) {
            auto [updated_value, changed] = ivalue_to(item, device, dtype, non_blocking);
            all_changed &= changed;
            some_changed |= changed;
            updated.emplace_back(std::move(updated_value));
        }
        if (some_changed) {
            if (!all_changed) {
                C10_THROW_ERROR(ValueError,
                    "list containing both metatensor and non-metatensor data "
                    "are not supported"
                );
            }
            return std::make_pair(updated, true);
        }
    } else if (ivalue.isTuple()) {
        const auto& tuple = ivalue.toTupleRef().elements();
        if (tuple.empty()) {
            return std::make_pair(ivalue, true);
        }

        auto updated = std::vector<torch::IValue>();
        auto some_changed = false;
        for (const auto& item: tuple) {
            auto [updated_value, changed] = ivalue_to(item, device, dtype, non_blocking);
            some_changed |= changed;
            updated.emplace_back(std::move(updated_value));
        }
        if (some_changed) {
            return std::make_pair(c10::ivalue::Tuple::create(updated), true);
        }
    }

    return std::make_pair(ivalue, false);
}

void Module::to(at::Device device, at::ScalarType dtype, bool non_blocking) {
    torch::jit::Module::to(device, dtype, non_blocking);
    this->to_impl_(device, dtype, non_blocking);
}

void Module::to(at::ScalarType dtype, bool non_blocking) {
    torch::jit::Module::to(dtype, non_blocking);
    this->to_impl_(torch::nullopt, dtype, non_blocking);
}

void Module::to(at::Device device, bool non_blocking) {
    torch::jit::Module::to(device, non_blocking);
    this->to_impl_(device, torch::nullopt, non_blocking);
}

void Module::to_impl_(
    const torch::optional<at::Device>& device,
    const torch::optional<at::ScalarType>& dtype,
    bool non_blocking
) {
    for (auto module: this->modules()) {
        for (const auto& attr: module.named_attributes(/*recurse=*/false)) {
            auto [value, changed] = ivalue_to(attr.value, device, dtype, non_blocking);
            if (changed) {
                module.register_attribute(attr.name, attr.value.type().get(), value);
            }
        }
    }
}
