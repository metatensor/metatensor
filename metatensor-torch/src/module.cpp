#include <string>
#include <vector>

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

static bool is_empty_container(const torch::IValue& ivalue) {
    if (ivalue.isGenericDict()) {
        return ivalue.toGenericDict().empty();
    } else if (ivalue.isList()) {
        return ivalue.toList().empty();
    } else if (ivalue.isTuple()) {
        return ivalue.toTupleRef().elements().empty();
    }
    return false;
}

// Convert metatensor values in an IValue tree to the given device/dtype.
//
// `legacy` is true for modules exported without `_mts_buffer_names`
// (metatensor-learn<0.6). On that path Module::to walks every attribute and can
// hit mixed dicts/lists; convert metatensor entries and leave the rest,
// matching tuples. This flag will be removed in the future, when we no longer
// have to support these models.
static std::pair<torch::IValue, bool> ivalue_to(
    torch::IValue ivalue,
    const torch::optional<at::Device>& device,
    const torch::optional<at::ScalarType>& dtype,
    bool non_blocking,
    bool legacy
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
            return std::make_pair(ivalue, false);
        }

        auto updated = c10::impl::GenericDict(dict.keyType(), dict.valueType());
        auto all_changed = true;
        auto some_changed = false;
        for (const auto& item: dict) {
            auto inner = item.value();
            if (is_empty_container(inner)) {
                updated.insert(item.key(), inner);
                continue;
            }
            auto [updated_value, changed] = ivalue_to(inner, device, dtype, non_blocking, legacy);
            all_changed &= changed;
            some_changed |= changed;
            updated.insert(item.key(), updated_value);
        }
        if (some_changed) {
            if (!all_changed && !legacy) {
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
            return std::make_pair(ivalue, false);
        }

        auto updated = c10::impl::GenericList(list.elementType());
        auto all_changed = true;
        auto some_changed = false;
        for (const auto& item: list) {
            auto inner = torch::IValue(item);
            if (is_empty_container(inner)) {
                updated.emplace_back(inner);
                continue;
            }
            auto [updated_value, changed] = ivalue_to(inner, device, dtype, non_blocking, legacy);
            all_changed &= changed;
            some_changed |= changed;
            updated.emplace_back(std::move(updated_value));
        }
        if (some_changed) {
            if (!all_changed && !legacy) {
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
            return std::make_pair(ivalue, false);
        }

        auto updated = std::vector<torch::IValue>();
        auto some_changed = false;
        for (const auto& item: tuple) {
            auto [updated_value, changed] = ivalue_to(item, device, dtype, non_blocking, legacy);
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
        // Determine which attributes to process
        std::vector<std::string> attr_names;
        // Modules without _mts_buffer_names are exported from
        // metatensor-learn<0.6. We allow dict/list to contain both metatensor
        // and non-metatensor data in this case.
        bool legacy = false;

        if (module.hasattr("_mts_buffer_names")) {
            auto names = module.attr("_mts_buffer_names").toList();
            for (const auto& item : names) {
                attr_names.push_back(item.get().toStringRef());
            }
        } else {
            // Fallback: process all attributes (backward compatibility with
            // modules that don't use register_buffer yet)
            legacy = true;
            TORCH_WARN_ONCE(
                "module does not have '_mts_buffer_names'; falling back to "
                "processing all attributes. This is deprecated and will be "
                "removed in a future version, update your version of "
                "metatensor-learn and use `register_buffer` explicitly "
                "to remove this warning."
            );
            for (const auto& attr : module.named_attributes(false)) {
                attr_names.push_back(attr.name);
            }
        }

        for (const auto& name : attr_names) {
            auto value = module.attr(name);
            auto [updated, changed] = ivalue_to(value, device, dtype, non_blocking, legacy);
            if (changed) {
                module.register_attribute(name, updated.type().get(), updated);
            }
        }
    }
}
