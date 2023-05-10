#include <torch/script.h>

#include "equistore.hpp"

#include "equistore/torch/labels.hpp"
#include "equistore/torch/block.hpp"
#include "equistore/torch/tensor.hpp"

using namespace equistore_torch;


// There is no way to access the docstrings from Python, so we don't bother
// setting them to something useful.
const std::string DOCSTRING = "";


static TorchLabelsEntry labels_entry(const TorchLabels& self, int64_t index) {
    return torch::make_intrusive<LabelsEntryHolder>(self, index);
}

// this function can not be implemented as a member of LabelsHolder, since it
// needs to receive a `TorchLabels` to give it to the `LabelsEntryHolder`
// constructor.
static torch::IValue labels_getitem(const TorchLabels& self, torch::IValue index) {
    if (index.isInt()) {
        return labels_entry(self, index.toInt());
    } else {
        auto names = equistore_torch::details::normalize_names(std::move(index), "index");
        return LabelsHolder::view(self, std::move(names));
    }
}


TORCH_LIBRARY(equistore, m) {
    m.class_<LabelsEntryHolder>("LabelsEntry")
        .def("__str__", &LabelsEntryHolder::__repr__)
        .def("__repr__", &LabelsEntryHolder::__repr__)
        .def("__len__", &LabelsEntryHolder::size)
        .def("__getitem__", &LabelsEntryHolder::getitem, DOCSTRING,
            {torch::arg("index")}
        )
        .def("__eq__", [](const TorchLabelsEntry& self, const TorchLabelsEntry& other){ return *self == *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def("__ne__", [](const TorchLabelsEntry& self, const TorchLabelsEntry& other){ return *self != *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def_property("names", &LabelsEntryHolder::names)
        .def_property("values", &LabelsEntryHolder::values)
        ;

    m.class_<LabelsHolder>("Labels")
        .def(
            torch::init<torch::IValue, torch::Tensor>(), DOCSTRING,
            {torch::arg("names"), torch::arg("values")}
        )
        .def("__str__", &LabelsHolder::__str__)
        // __repr__ is ignored for now, until we can use
        // https://github.com/pytorch/pytorch/pull/100724 (hopefully torch 2.1)
        .def("__repr__", &LabelsHolder::__repr__)
        .def("__len__", &LabelsHolder::count)
        .def("__contains__", [](const TorchLabels& self, torch::IValue entry){
            return self->position(entry).has_value();
            }, DOCSTRING,
            {torch::arg("entry")}
        )
        .def("__eq__", [](const TorchLabels& self, const TorchLabels& other){ return *self == *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def("__ne__", [](const TorchLabels& self, const TorchLabels& other){ return *self != *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def("__getitem__", labels_getitem, DOCSTRING, {torch::arg("index")})
        .def_static("single", &LabelsHolder::single)
        .def_static("empty", &LabelsHolder::empty)
        .def_static("range", &LabelsHolder::range)
        .def("entry", labels_entry, DOCSTRING, {torch::arg("index")})
        .def("view", [](const TorchLabels& self, torch::IValue names) {
            auto names_vector = equistore_torch::details::normalize_names(std::move(names), "names");
            return LabelsHolder::view(self, std::move(names_vector));
        }, DOCSTRING, {torch::arg("names")})
        .def_property("names", &LabelsHolder::names)
        .def_property("values", &LabelsHolder::values)
        .def("position", &LabelsHolder::position, DOCSTRING,
            {torch::arg("entry")}
        )
        .def("print", &LabelsHolder::print, DOCSTRING,
            {torch::arg("max_entries"), torch::arg("indent") = 0}
        )
        .def("is_view", &LabelsHolder::is_view)
        .def("to_owned", [](const TorchLabels& self){ return torch::make_intrusive<LabelsHolder>(self->to_owned()); })
        ;

    m.class_<TensorBlockHolder>("TensorBlock")
        .def(
            torch::init<torch::Tensor, TorchLabels, std::vector<TorchLabels>, TorchLabels>(), DOCSTRING,
            {torch::arg("values"), torch::arg("samples"), torch::arg("components"), torch::arg("properties")}
        )
        .def("copy", &TensorBlockHolder::copy)
        .def_property("values", &TensorBlockHolder::values)
        .def_property("samples", &TensorBlockHolder::samples)
        .def_property("components", &TensorBlockHolder::components)
        .def_property("properties", &TensorBlockHolder::properties)
        .def("add_gradient", &TensorBlockHolder::add_gradient, DOCSTRING,
            {torch::arg("parameter"), torch::arg("gradient")}
        )
        .def("gradients_list", &TensorBlockHolder::gradients_list)
        .def("has_gradient", &TensorBlockHolder::has_gradient, DOCSTRING,
            {torch::arg("parameter")}
        )
        .def("gradient", &TensorBlockHolder::gradient, DOCSTRING,
            {torch::arg("parameter")}
        )
        .def("gradients", &TensorBlockHolder::gradients)
        ;

    m.class_<TensorMapHolder>("TensorMap")
        .def(
            torch::init<TorchLabels, std::vector<TorchTensorBlock>>(), DOCSTRING,
            {torch::arg("keys"), torch::arg("blocks")}
        )
        .def("__len__", [](const TorchTensorMap& tensor){ return tensor->keys()->count(); })
        .def("copy", &TensorMapHolder::copy)
        .def_property("keys", &TensorMapHolder::keys)
        .def("blocks_matching", &TensorMapHolder::blocks_matching, DOCSTRING,
            {torch::arg("selection")}
        )
        .def("block_by_id", &TensorMapHolder::block_by_id, DOCSTRING,
            {torch::arg("index")}
        )
        .def("keys_to_samples", &TensorMapHolder::keys_to_samples, DOCSTRING,
            {torch::arg("keys_to_move"), torch::arg("sort_samples") = true}
        )
        .def("keys_to_properties", &TensorMapHolder::keys_to_properties, DOCSTRING,
            {torch::arg("keys_to_move"), torch::arg("sort_samples") = true}
        )
        .def("components_to_properties", &TensorMapHolder::components_to_properties, DOCSTRING,
            {torch::arg("dimensions")}
        )
        .def_property("sample_names", &TensorMapHolder::sample_names)
        .def_property("components_names", &TensorMapHolder::components_names)
        .def_property("property_names", &TensorMapHolder::property_names)
        // TODO
        // .def_pickle(
        //     // __getstate__
        //     [](const torch::intrusive_ptr<TorchTensorMap>& self) -> std::string {
        //          // TODO
        //     },
        //     // __setstate__
        //     [](std::string state) -> torch::intrusive_ptr<TorchCalculator> {
        //         // TODO
        //     })
        ;
}
