#include <torch/script.h>

#include "metatensor/torch/labels.hpp"
#include "metatensor/torch/block.hpp"
#include "metatensor/torch/tensor.hpp"
#include "metatensor/torch/misc.hpp"

#include "./utils.hpp"

using namespace metatensor_torch;


static LabelsEntry labels_entry(const Labels& self, int64_t index) {
    return torch::make_intrusive<LabelsEntryHolder>(self, index);
}

// this function can not be implemented as a member of LabelsHolder, since it
// needs to receive a `Labels` to give it to the `LabelsEntryHolder`
// constructor.
static torch::IValue labels_getitem(const Labels& self, torch::IValue index) {
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

template <typename T>
bool custom_class_is(torch::IValue ivalue) {
    assert(ivalue.isCustomClass());

    // this is inspired by the code inside `torch::IValue.toCustomClass<T>()`
    auto* expected_type = torch::getCustomClassType<torch::intrusive_ptr<T>>().get();
    return ivalue.type().get() == expected_type;
}

static void save_ivalue(const std::string& path, torch::IValue data) {
    if (data.isCustomClass()) {
        if (custom_class_is<TensorMapHolder>(data)) {
            auto tensor = data.toCustomClass<TensorMapHolder>();
            metatensor_torch::save(path, tensor);
            return;
        } else if (custom_class_is<TensorBlockHolder>(data)) {
            auto block = data.toCustomClass<TensorBlockHolder>();
            metatensor_torch::save(path, block);
            return;
        } else if (custom_class_is<LabelsHolder>(data)) {
            auto labels = data.toCustomClass<LabelsHolder>();
            metatensor_torch::save(path, labels);
            return;
        }
    }

    C10_THROW_ERROR(TypeError,
        "`data` must be one of 'Labels', 'TensorBlock' or 'TensorMap' in `save`, "
        "not " + data.type()->str()
    );
}

static torch::Tensor save_ivalue_buffer(torch::IValue data) {
    if (data.isCustomClass()) {
        if (custom_class_is<TensorMapHolder>(data)) {
            auto tensor = data.toCustomClass<TensorMapHolder>();
            return metatensor_torch::save_buffer(tensor);
        } else if (custom_class_is<TensorBlockHolder>(data)) {
            auto block = data.toCustomClass<TensorBlockHolder>();
            return metatensor_torch::save_buffer(block);
        } else if (custom_class_is<LabelsHolder>(data)) {
            auto labels = data.toCustomClass<LabelsHolder>();
            return metatensor_torch::save_buffer(labels);
        }
    }

    C10_THROW_ERROR(TypeError,
        "`data` must be one of 'Labels', 'TensorBlock' or 'TensorMap' in `save_buffer`, "
        "not " + data.type()->str()
    );
}

static void block_values_setter(TensorBlock, torch::Tensor) {
    C10_THROW_ERROR(
        ValueError,
        "Direct assignment to `values` is not possible. "
        "Please use `block.values[:] = new_values` instead."
    );
}


TORCH_LIBRARY(metatensor, m) {
    // There is no way to access the docstrings from Python, so we don't bother
    // setting them to something useful here.
    //
    // Whenever this file is changed, please also reproduce the changes in
    // python/metatensor_torch/metatensor/torch/documentation.py, and include the
    // docstring over there
    const std::string DOCSTRING;


    m.class_<LabelsEntryHolder>("LabelsEntry")
        .def("__str__", &LabelsEntryHolder::repr)
        .def("__repr__", &LabelsEntryHolder::repr)
        .def("__len__", &LabelsEntryHolder::size)
        .def("__getitem__", &LabelsEntryHolder::getitem, DOCSTRING,
            {torch::arg("index")}
        )
        .def("__eq__", [](const LabelsEntry& self, const LabelsEntry& other){ return *self == *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def("__ne__", [](const LabelsEntry& self, const LabelsEntry& other){ return *self != *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def_property("names", &LabelsEntryHolder::names)
        .def_property("values", &LabelsEntryHolder::values)
        .def_property("device", &LabelsEntryHolder::device)
        .def("print", &LabelsEntryHolder::print)
        ;

    m.class_<LabelsHolder>("Labels")
        .def(
            torch::init([](torch::IValue names, torch::Tensor values, bool assume_unique) {
                if (assume_unique) {
                    return c10::make_intrusive<LabelsHolder>(std::
                        move(names), std::move(values), metatensor::assume_unique{}
                    );
                } else {
                    return c10::make_intrusive<LabelsHolder>(
                        std::move(names), std::move(values)
                    );
                }
            }),
            DOCSTRING,
            {torch::arg("names"), torch::arg("values"), torch::arg("assume_unique") = false}
        )
        .def("__str__", &LabelsHolder::str)
        .def("__repr__", &LabelsHolder::repr)
        .def("__len__", &LabelsHolder::count)
        .def("__contains__", [](const Labels& self, torch::IValue entry) {
                return self->position(entry).has_value();
            }, DOCSTRING,
            {torch::arg("entry")}
        )
        .def("__eq__", [](const Labels& self, const Labels& other){ return *self == *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def("__ne__", [](const Labels& self, const Labels& other){ return *self != *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def("__getitem__", labels_getitem, DOCSTRING, {torch::arg("index")})
        .def_static("single", &LabelsHolder::single)
        .def_static("empty", &LabelsHolder::empty)
        .def_static("range", &LabelsHolder::range)
        .def("save", &LabelsHolder::save, DOCSTRING, {torch::arg("file")})
        .def("save_buffer", &LabelsHolder::save_buffer)
        .def_static("load", &LabelsHolder::load)
        .def_static("load_buffer", &LabelsHolder::load_buffer)
        .def("entry", labels_entry, DOCSTRING, {torch::arg("index")})
        .def("column", &LabelsHolder::column, DOCSTRING, {torch::arg("dimension")})
        .def("view", [](const Labels& self, torch::IValue names) {
            auto names_vector = metatensor_torch::details::normalize_names(std::move(names), "names");
            return LabelsHolder::view(self, std::move(names_vector));
        }, DOCSTRING, {torch::arg("names")})
        .def_property("names", &LabelsHolder::names)
        .def_property("values", &LabelsHolder::values)
        .def("append", &LabelsHolder::append, DOCSTRING, {torch::arg("name"), torch::arg("values")})
        .def("insert", &LabelsHolder::insert, DOCSTRING, {torch::arg("index"), torch::arg("name"), torch::arg("values")})
        .def("permute", &LabelsHolder::permute, DOCSTRING, {torch::arg("dimensions_indexes")})
        .def("remove", &LabelsHolder::remove, DOCSTRING, {torch::arg("name")})
        .def("rename", &LabelsHolder::rename, DOCSTRING, {torch::arg("old"), torch::arg("new")})
        .def("to",
            static_cast<Labels (LabelsHolder::*)(torch::IValue, bool) const>(&LabelsHolder::to),
            DOCSTRING, {torch::arg("device"), torch::arg("non_blocking") = false}
        )
        .def_property("device", &LabelsHolder::device)
        .def("position", &LabelsHolder::position, DOCSTRING,
            {torch::arg("entry")}
        )
        .def("print", &LabelsHolder::print, DOCSTRING,
            {torch::arg("max_entries"), torch::arg("indent") = 0}
        )
        .def("is_view", &LabelsHolder::is_view)
        .def("to_owned", &LabelsHolder::to_owned)
        .def("union", &LabelsHolder::set_union, DOCSTRING, {torch::arg("other")})
        .def("union_and_mapping", &LabelsHolder::union_and_mapping, DOCSTRING, {torch::arg("other")})
        .def("intersection", &LabelsHolder::set_intersection, DOCSTRING, {torch::arg("other")})
        .def("intersection_and_mapping", &LabelsHolder::intersection_and_mapping, DOCSTRING, {torch::arg("other")})
        .def("difference", &LabelsHolder::set_difference, DOCSTRING, {torch::arg("other")})
        .def("difference_and_mapping", &LabelsHolder::difference_and_mapping, DOCSTRING, {torch::arg("other")})
        .def("select", &LabelsHolder::select, DOCSTRING, {torch::arg("selection")})
        .def_pickle(
            // __getstate__
            [](const Labels& self){ return self->save_buffer(); },
            // __setstate__
            [](torch::Tensor buffer){ return metatensor_torch::load_labels_buffer(buffer); }
        );


    m.class_<TensorBlockHolder>("TensorBlock")
        .def(
            torch::init<torch::Tensor, Labels, std::vector<Labels>, Labels>(), DOCSTRING,
            {torch::arg("values"), torch::arg("samples"), torch::arg("components"), torch::arg("properties")}
        )
        .def("__repr__", &TensorBlockHolder::repr)
        .def("__str__", &TensorBlockHolder::repr)
        .def("__len__", &TensorBlockHolder::len )
        .def("copy", &TensorBlockHolder::copy)
        .def_property("values", &TensorBlockHolder::values, block_values_setter)
        .def_property("samples", &TensorBlockHolder::samples)
        .def_property("components", &TensorBlockHolder::components)
        .def_property("properties", &TensorBlockHolder::properties)
        .def_property("shape", &TensorBlockHolder::shape )
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
        .def_property("device", &TensorBlockHolder::device)
        .def_property("dtype", &TensorBlockHolder::scalar_type)
        .def("to", &TensorBlockHolder::to_positional, DOCSTRING, {
            torch::arg("_0") = torch::IValue(),
            torch::arg("_1") = torch::IValue(),
            torch::arg("dtype") = torch::nullopt,
            torch::arg("device") = torch::nullopt,
            torch::arg("arrays") = torch::nullopt,
            torch::arg("non_blocking") = false
        })
        .def("save", &TensorBlockHolder::save, DOCSTRING, {torch::arg("file")})
        .def("save_buffer", &TensorBlockHolder::save_buffer)
        .def_static("load", &TensorBlockHolder::load)
        .def_static("load_buffer", &TensorBlockHolder::load_buffer)
        .def_pickle(
            // __getstate__
            [](const TensorBlock& self){ return self->save_buffer(); },
            // __setstate__
            [](torch::Tensor buffer){ return metatensor_torch::load_block_buffer(buffer); }
        );

    m.class_<TensorMapHolder>("TensorMap")
        .def(
            torch::init<Labels, std::vector<TensorBlock>>(), DOCSTRING,
            {torch::arg("keys"), torch::arg("blocks")}
        )
        .def("__len__", [](const TensorMap& self){ return self->keys()->count(); })
        .def("__repr__", [](const TensorMap& self){ return self->print(-1); })
        .def("__str__", [](const TensorMap& self){ return self->print(4); })
        .def("__getitem__", &TensorMapHolder::block_torch, DOCSTRING,
            {torch::arg("selection")}
        )
        .def("copy", &TensorMapHolder::copy)
        .def("save", &TensorMapHolder::save, DOCSTRING, {torch::arg("file")})
        .def("save_buffer", &TensorMapHolder::save_buffer)
        .def_static("load", &TensorMapHolder::load)
        .def_static("load_buffer", &TensorMapHolder::load_buffer)
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
        .def_property("component_names", &TensorMapHolder::component_names)
        .def_property("property_names", &TensorMapHolder::property_names)
        .def_property("device", &TensorMapHolder::device)
        .def_property("dtype", &TensorMapHolder::scalar_type)
        .def("to", &TensorMapHolder::to_positional, DOCSTRING, {
            torch::arg("_0") = torch::IValue(),
            torch::arg("_1") = torch::IValue(),
            torch::arg("dtype") = torch::nullopt,
            torch::arg("device") = torch::nullopt,
            torch::arg("arrays") = torch::nullopt,
            torch::arg("non_blocking") = false
        })
        .def("print", &TensorMapHolder::print, DOCSTRING,
            {torch::arg("max_keys")}
        )
        .def_pickle(
            // __getstate__
            [](const TensorMap& self){ return self->save_buffer(); },
            // __setstate__
            [](torch::Tensor buffer){ return metatensor_torch::load_buffer(buffer); }
        );


    // standalone functions
    m.def("version() -> str", metatensor_torch::version);
    m.def("dtype_name(ScalarType dtype) -> str", scalar_type_name);

    m.def(
        "load(str file) -> __torch__.torch.classes.metatensor.TensorMap",
        metatensor_torch::load
    );
    m.def(
        "load_buffer(Tensor buffer) -> __torch__.torch.classes.metatensor.TensorMap",
        metatensor_torch::load_buffer
    );

    m.def(
        "load_block(str file) -> __torch__.torch.classes.metatensor.TensorBlock",
        metatensor_torch::load_block
    );
    m.def(
        "load_block_buffer(Tensor buffer) -> __torch__.torch.classes.metatensor.TensorBlock",
        metatensor_torch::load_block_buffer
    );

    m.def(
        "load_labels(str file) -> __torch__.torch.classes.metatensor.Labels",
        metatensor_torch::load_labels
    );
    m.def(
        "load_labels_buffer(Tensor buffer) -> __torch__.torch.classes.metatensor.Labels",
        metatensor_torch::load_labels_buffer
    );

    // manually construct the schema for "save(str file, Any data) -> ()", so we
    // can set AliasAnalysisKind to CONSERVATIVE. In turn, this make it so
    // the TorchScript compiler knows this function has side-effects, and does
    // not remove it from the graph.
    auto schema = c10::FunctionSchema(
        /*name=*/"save",
        /*overload_name=*/"save",
        /*arguments=*/{
            c10::Argument("file", c10::getTypePtr<std::string>()),
            c10::Argument("data", c10::getTypePtr<c10::IValue>())
        },
        /*returns=*/{}
    );
    schema.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
    m.def(std::move(schema), save_ivalue);

    m.def("save_buffer(Any data) -> Tensor", save_ivalue_buffer);
}
