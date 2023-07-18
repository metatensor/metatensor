#include <torch/script.h>

#include "equistore.hpp"

#include "equistore/torch/labels.hpp"
#include "equistore/torch/block.hpp"
#include "equistore/torch/tensor.hpp"
#include "equistore/torch/misc.hpp"

using namespace equistore_torch;


static TorchLabelsEntry labels_entry(const TorchLabels& self, int64_t index) {
    return torch::make_intrusive<LabelsEntryHolder>(self, index);
}

// this function can not be implemented as a member of LabelsHolder, since it
// needs to receive a `TorchLabels` to give it to the `LabelsEntryHolder`
// constructor.
static torch::IValue labels_getitem(const TorchLabels& self, torch::IValue index) {
    if (index.isInt()) {
        return labels_entry(self, index.toInt());
    } else if (index.isString()) {
        return self->column(index.toStringRef());
    } else {
        C10_THROW_ERROR(TypeError,
            "Labels can only be indexed by int or str, got '" + index.type()->str() + "' instead"
        );
    }
}


TORCH_LIBRARY(equistore, m) {
    // There is no way to access the docstrings from Python, so we don't bother
    // setting them to something useful here.
    //
    // Whenever this file is changed, please also reproduce the changes in
    // python/equistore-torch/equistore/torch/documentation.py, and include the
    // docstring over there
    const std::string DOCSTRING = "";


    m.class_<LabelsEntryHolder>("LabelsEntry")
        .def("__str__", &LabelsEntryHolder::__repr__)
        .def("__repr__", &LabelsEntryHolder::__repr__)
        .def("__len__", &LabelsEntryHolder::size)
        .def("__getitem__", &LabelsEntryHolder::__getitem__, DOCSTRING,
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
        .def("print", &LabelsEntryHolder::print)
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
        .def("__contains__", [](const TorchLabels& self, torch::IValue entry) {
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
        .def("column", &LabelsHolder::column, DOCSTRING, {torch::arg("dimension")})
        .def("view", [](const TorchLabels& self, torch::IValue names) {
            auto names_vector = equistore_torch::details::normalize_names(std::move(names), "names");
            return LabelsHolder::view(self, std::move(names_vector));
        }, DOCSTRING, {torch::arg("names")})
        .def_property("names", &LabelsHolder::names)
        .def_property("values", &LabelsHolder::values)
        .def("to", &LabelsHolder::to, DOCSTRING,
            {torch::arg("device")}
        )
        .def("position", &LabelsHolder::position, DOCSTRING,
            {torch::arg("entry")}
        )
        .def("print", &LabelsHolder::print, DOCSTRING,
            {torch::arg("max_entries"), torch::arg("indent") = 0}
        )
        .def("is_view", &LabelsHolder::is_view)
        .def("to_owned", [](const TorchLabels& self){ return torch::make_intrusive<LabelsHolder>(self->to_owned()); })
        .def("union", &LabelsHolder::set_union, DOCSTRING, {torch::arg("other")})
        .def("union_and_mapping", &LabelsHolder::union_and_mapping, DOCSTRING, {torch::arg("other")})
        .def("intersection", &LabelsHolder::set_intersection, DOCSTRING, {torch::arg("other")})
        .def("intersection_and_mapping", &LabelsHolder::intersection_and_mapping, DOCSTRING, {torch::arg("other")})
        ;

    m.class_<TensorBlockHolder>("TensorBlock")
        .def(
            torch::init<torch::Tensor, TorchLabels, std::vector<TorchLabels>, TorchLabels>(), DOCSTRING,
            {torch::arg("values"), torch::arg("samples"), torch::arg("components"), torch::arg("properties")}
        )
        .def("__repr__", &TensorBlockHolder::__repr__)
        .def("__str__", &TensorBlockHolder::__repr__)
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
        .def("__len__", [](const TorchTensorMap& self){ return self->keys()->count(); })
        .def("__repr__", [](const TorchTensorMap& self){ return self->print(-1); })
        .def("__str__", [](const TorchTensorMap& self){ return self->print(4); })
        .def("__getitem__", &TensorMapHolder::block_torch, DOCSTRING,
            {torch::arg("selection")}
        )
        .def("copy", &TensorMapHolder::copy)
        .def("items", &TensorMapHolder::items)
        .def_property("keys", &TensorMapHolder::keys)
        .def("blocks_matching", &TensorMapHolder::blocks_matching, DOCSTRING,
            {torch::arg("selection")}
        )
        .def("block_by_id", &TensorMapHolder::block_by_id, DOCSTRING,
            {torch::arg("index")}
        )
        .def("blocks_by_id", &TensorMapHolder::blocks_by_id, DOCSTRING,
            {torch::arg("indices")}
        )
        .def("block", &TensorMapHolder::block_torch, DOCSTRING,
            {torch::arg("selection") = torch::IValue()}
        )
        .def("blocks", &TensorMapHolder::blocks_torch, DOCSTRING,
            {torch::arg("selection") = torch::IValue()}
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
        .def("print", &TensorMapHolder::print, DOCSTRING,
            {torch::arg("max_keys")}
        )
        .def_pickle(
            // __getstate__
            [](const TorchTensorMap& self) -> torch::Tensor {
                auto buffer = equistore::TensorMap::save_buffer(self->as_equistore());
                // move the buffer to the heap so it can escape this function
                // `torch::from_blob` does not take ownership of the data,
                // so we need to register a custom deleter to clean up when
                // the tensor is no longer used
                auto buffer_data = new std::vector<uint8_t>(std::move(buffer));

                auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCPU);
                auto deleter = [=](void* data) {
                    delete buffer_data;
                };

                // use a tensor of bytes to store the data
                return torch::from_blob(
                    buffer_data->data(),
                    {static_cast<int64_t>(buffer_data->size())},
                    deleter,
                    options
                );
            },
            // __setstate__
            [](torch::Tensor buffer) -> TorchTensorMap {
                return torch::make_intrusive<TensorMapHolder>(
                    equistore::TensorMap::load_buffer(
                        buffer.data_ptr<uint8_t>(),
                        buffer.size(0),
                        details::create_torch_array
                    )
                );
            })
        ;

    m.def("load", equistore_torch::load);
    m.def("save", equistore_torch::save);
}
