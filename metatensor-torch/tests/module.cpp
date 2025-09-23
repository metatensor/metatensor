#include <torch/torch.h>

#include "metatensor/torch.hpp"

#include <catch.hpp>

static void check_dtype_device(const torch::jit::Module& module, torch::Device device, torch::ScalarType scalar_type) {
    std::vector<std::string> all_fields = {};
    for (const auto& item: module.named_attributes()) {
        if (item.name.find("_is_full_backward_hook") != std::string::npos) {
            continue;
        }
        if (item.name.find("training") != std::string::npos) {
            continue;
        }

        all_fields.push_back(item.name);
        if (item.name == "labels" || item.name == "a.labels") {
            auto labels = item.value.toCustomClass<metatensor_torch::LabelsHolder>();
            CHECK(labels->device() == device);
            CHECK(labels->values().scalar_type() == torch::kInt32);
        }

        if (item.name == "block" || item.name == "b.block") {
            auto block = item.value.toCustomClass<metatensor_torch::TensorBlockHolder>();
            CHECK(block->device() == device);
            CHECK(block->scalar_type() == scalar_type);
        }

        if (item.name == "tensor" || item.name == "c.tensor") {
            auto tensor = item.value.toCustomClass<metatensor_torch::TensorMapHolder>();
            CHECK(tensor->device() == device);
            CHECK(tensor->scalar_type() == scalar_type);
        }

        if (item.name == "tuple") {
            auto tuple = item.value.toTuple()->elements();
            auto labels = tuple[0].toCustomClass<metatensor_torch::LabelsHolder>();
            CHECK(labels->device() == device);
            CHECK(labels->values().scalar_type() == torch::kInt32);

            auto block = tuple[1].toCustomClass<metatensor_torch::TensorBlockHolder>();
            CHECK(block->device() == device);
            CHECK(block->scalar_type() == scalar_type);

            auto tensor = tuple[2].toCustomClass<metatensor_torch::TensorMapHolder>();
            CHECK(tensor->device() == device);
            CHECK(tensor->scalar_type() == scalar_type);
        }

        if (item.name == "a.dict") {
            auto dict = item.value.toGenericDict();
            auto labels = dict.at("labels").toCustomClass<metatensor_torch::LabelsHolder>();
            CHECK(labels->device() == device);
            CHECK(labels->values().scalar_type() == torch::kInt32);
        }

        if (item.name == "a.list") {
            auto list = item.value.toList();
            auto labels = list.get(0).toCustomClass<metatensor_torch::LabelsHolder>();
            CHECK(labels->device() == device);
            CHECK(labels->values().scalar_type() == torch::kInt32);
        }

        if (item.name == "a.tuple") {
            auto tuple = item.value.toTuple()->elements();
            auto labels = tuple[0].toCustomClass<metatensor_torch::LabelsHolder>();
            CHECK(labels->device() == device);
            CHECK(labels->values().scalar_type() == torch::kInt32);
        }

        if (item.name == "a.nested") {
            auto str_dict = item.value.toGenericDict();
            auto int_dict = str_dict.at("dict").toGenericDict();
            auto first_list = int_dict.at(42).toList();
            auto second_list = first_list.get(0).toList();

            auto labels = second_list.get(0).toCustomClass<metatensor_torch::LabelsHolder>();
            CHECK(labels->device() == device);
            CHECK(labels->values().scalar_type() == torch::kInt32);

            CHECK(first_list.get(1).toList().empty());
            CHECK(int_dict.at(50).toList().empty());
            CHECK(str_dict.at("empty").toGenericDict().empty());
        }

        if (item.name == "b.dict") {
            auto dict = item.value.toGenericDict();
            auto block = dict.at("block").toCustomClass<metatensor_torch::TensorBlockHolder>();
            CHECK(block->device() == device);
            CHECK(block->scalar_type() == scalar_type);
        }

        if (item.name == "b.list") {
            auto list = item.value.toList();
            auto block = list.get(0).toCustomClass<metatensor_torch::TensorBlockHolder>();
            CHECK(block->device() == device);
            CHECK(block->scalar_type() == scalar_type);
        }

        if (item.name == "b.tuple") {
            auto tuple = item.value.toTuple()->elements();
            auto block = tuple[0].toCustomClass<metatensor_torch::TensorBlockHolder>();
            CHECK(block->device() == device);
            CHECK(block->scalar_type() == scalar_type);
        }

        if (item.name == "b.nested") {
            auto str_dict = item.value.toGenericDict();
            auto int_dict = str_dict.at("dict").toGenericDict();
            auto first_list = int_dict.at(42).toList();
            auto second_list = first_list.get(0).toList();

            auto block = second_list.get(0).toCustomClass<metatensor_torch::TensorBlockHolder>();
            CHECK(block->device() == device);
            CHECK(block->scalar_type() == scalar_type);

            CHECK(first_list.get(1).toList().empty());
            CHECK(int_dict.at(50).toList().empty());
            CHECK(str_dict.at("empty").toGenericDict().empty());
        }

        if (item.name == "c.dict") {
            auto dict = item.value.toGenericDict();
            auto tensor = dict.at("tensor").toCustomClass<metatensor_torch::TensorMapHolder>();
            CHECK(tensor->device() == device);
            CHECK(tensor->scalar_type() == scalar_type);
        }

        if (item.name == "c.list") {
            auto list = item.value.toList();
            auto tensor = list.get(0).toCustomClass<metatensor_torch::TensorMapHolder>();
            CHECK(tensor->device() == device);
            CHECK(tensor->scalar_type() == scalar_type);
        }

        if (item.name == "c.tuple") {
            auto tuple = item.value.toTuple()->elements();
            auto tensor = tuple[0].toCustomClass<metatensor_torch::TensorMapHolder>();
            CHECK(tensor->device() == device);
            CHECK(tensor->scalar_type() == scalar_type);
        }

        if (item.name == "c.nested") {
            auto str_dict = item.value.toGenericDict();
            auto int_dict = str_dict.at("dict").toGenericDict();
            auto first_list = int_dict.at(42).toList();
            auto second_list = first_list.get(0).toList();

            auto tensor = second_list.get(0).toCustomClass<metatensor_torch::TensorMapHolder>();
            CHECK(tensor->device() == device);
            CHECK(tensor->scalar_type() == scalar_type);

            CHECK(first_list.get(1).toList().empty());
            CHECK(int_dict.at(50).toList().empty());
            CHECK(str_dict.at("empty").toGenericDict().empty());
        }
    }

    std::vector<std::string> EXPECTED_FIELDS = {
        "a", "a.dict", "a.labels", "a.list", "a.nested", "a.tuple",
        "b", "b.block", "b.dict", "b.list", "b.nested", "b.tuple",
        "block",
        "c", "c.dict", "c.list", "c.nested", "c.tensor", "c.tuple",
        "labels", "tensor", "tuple"
    };

    std::sort(std::begin(all_fields), std::end(all_fields));
    CHECK(all_fields == EXPECTED_FIELDS);
}


TEST_CASE("Module") {
    auto module = torch::jit::load(TEST_TORCH_SCRIPT_MODULE);

    auto mts_module = metatensor_torch::Module(module);
    check_dtype_device(mts_module, torch::Device("cpu"), torch::kFloat64);

    mts_module.to(torch::kFloat32);
    check_dtype_device(mts_module, torch::Device("cpu"), torch::kFloat32);

    mts_module.to(torch::Device("meta"));
    check_dtype_device(mts_module, torch::Device("meta"), torch::kFloat32);
}
