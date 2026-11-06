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

// `array_backend` is used by metatensor-core to decide which array type to use
// for data not allocated by Python. metatensor-torch always uses torch.Tensor,
// and only accepts this parameter for compatibility with metatensor-core.
static void check_array_backend(const std::string& array_backend) {
    if (array_backend != "torch") {
        C10_THROW_ERROR(ValueError,
            "invalid `array_backend` '" + array_backend + "', only 'torch' is "
            "supported by metatensor-torch"
        );
    }
}

// `to()` used to take an `arrays` parameter, which is now called
// `array_backend`. This handles the deprecated parameter name.
static torch::optional<std::string> to_array_backend(
    const std::string& context,
    torch::optional<std::string> array_backend,
    torch::optional<std::string> arrays
) {
    if (arrays.has_value()) {
        if (array_backend.has_value()) {
            C10_THROW_ERROR(ValueError,
                "can not give both `arrays` and `array_backend` in " + context
            );
        }

        TORCH_WARN_DEPRECATION(
            "the `arrays` parameter of " + context + " is deprecated, "
            "please use `array_backend` instead"
        );
        return arrays;
    }
    return array_backend;
}

// Register a static method with named arguments and default values. This is
// supported by `torch::class_::def_static` since torch 2.13, and this function
// contains a backport of the corresponding code for older versions of torch.
template <typename Class, typename Func>
static void def_static(
    torch::class_<Class>& klass,
    std::string name,
    Func func,
    std::string docstring,
    std::initializer_list<torch::arg> args
) {
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 13)
    klass.def_static(std::move(name), std::move(func), std::move(docstring), args);
#else
    // the class type is available through `getCustomClassType` instead
    (void)klass;

    const auto& class_type = c10::getCustomClassType<c10::intrusive_ptr<Class>>();
    auto qualified_name = class_type->name()->qualifiedName() + "." + name;

    auto schema = c10::inferFunctionSchemaSingleReturn<Func>(std::move(name), "");
    const auto& old_arguments = schema.arguments();
    TORCH_CHECK(
        args.size() == old_arguments.size(),
        "names must be specified for all arguments of ", qualified_name
    );

    auto arguments = std::vector<c10::Argument>();
    arguments.reserve(old_arguments.size());
    size_t i = 0;
    for (const auto& arg: args) {
        const auto& old_arg = old_arguments[i++];
        arguments.emplace_back(
            arg.name_,
            old_arg.type(),
            old_arg.real_type(),
            old_arg.N(),
            arg.value_
        );
    }
    schema = schema.cloneWithArguments(std::move(arguments));

    auto wrapped_func = [func = std::move(func)](torch::jit::Stack& stack) mutable -> void {
        using RetType = typename c10::guts::infer_function_traits_t<Func>::return_type;
        torch::detail::BoxedProxy<RetType, Func>()(stack, func);
    };
    auto method = std::make_unique<torch::jit::BuiltinOpFunction>(
        std::move(qualified_name),
        std::move(schema),
        std::move(wrapped_func),
        std::move(docstring)
    );

    class_type->addStaticMethod(method.get());
    torch::registerCustomClassMethod(std::move(method));
#endif
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

    auto labels_class = m.class_<LabelsHolder>("Labels");
    labels_class
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
        .def("save", &LabelsHolder::save, DOCSTRING, {torch::arg("file")})
        .def("save_buffer", &LabelsHolder::save_buffer)
        .def("release", [](Labels self) -> int64_t {
            const auto* ptr = self->release().release();
            return reinterpret_cast<int64_t>(ptr);
        })
        .def("as_mts_labels_t", [](const Labels& self) -> int64_t {
            const auto* ptr = self->as_metatensor().as_mts_labels_t();
            return reinterpret_cast<int64_t>(ptr);
        })
        .def("entry", labels_entry, DOCSTRING, {torch::arg("index")})
        .def("column", &LabelsHolder::column, DOCSTRING, {torch::arg("dimension")})
        .def_property("names", &LabelsHolder::names)
        .def_property("values", &LabelsHolder::values)
        .def("append", &LabelsHolder::append, DOCSTRING, {torch::arg("name"), torch::arg("values")})
        .def("insert", &LabelsHolder::insert, DOCSTRING, {torch::arg("index"), torch::arg("name"), torch::arg("values")})
        .def("permute", &LabelsHolder::permute, DOCSTRING, {torch::arg("dimensions_indexes")})
        .def("remove", &LabelsHolder::remove, DOCSTRING, {torch::arg("name")})
        .def("rename", &LabelsHolder::rename, DOCSTRING, {torch::arg("old"), torch::arg("new")})
        .def("to", [](const Labels& self, torch::IValue device, bool non_blocking, torch::optional<std::string> array_backend) {
                if (array_backend.value_or("torch") != "torch") {
                    C10_THROW_ERROR(ValueError,
                        "`array_backend` must be None or 'torch', got '" + array_backend.value() + "' instead"
                    );
                }
                return self->to(std::move(device), non_blocking);
            }, DOCSTRING,
            {torch::arg("device"), torch::arg("non_blocking") = false, torch::arg("array_backend") = torch::nullopt}
        )
        .def_property("device", &LabelsHolder::device)
        .def("position", &LabelsHolder::position, DOCSTRING,
            {torch::arg("entry")}
        )
        .def("print", &LabelsHolder::print, DOCSTRING,
            {torch::arg("max_entries"), torch::arg("indent") = 0}
        )
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

    def_static(labels_class, "single", [](const std::string& array_backend) {
            check_array_backend(array_backend);
            return LabelsHolder::single();
        }, DOCSTRING,
        {torch::arg("array_backend") = "torch"}
    );

    def_static(labels_class, "empty", [](torch::IValue names, const std::string& array_backend) {
            check_array_backend(array_backend);
            return LabelsHolder::empty(std::move(names));
        }, DOCSTRING,
        {torch::arg("names"), torch::arg("array_backend") = "torch"}
    );

    def_static(labels_class, "range", [](std::string name, int64_t end, const std::string& array_backend) {
            check_array_backend(array_backend);
            return LabelsHolder::range(std::move(name), end);
        }, DOCSTRING,
        {torch::arg("name"), torch::arg("end"), torch::arg("array_backend") = "torch"}
    );

    def_static(labels_class, "load", [](const std::string& file, const std::string& array_backend) {
            check_array_backend(array_backend);
            return LabelsHolder::load(file);
        }, DOCSTRING,
        {torch::arg("file"), torch::arg("array_backend") = "torch"}
    );

    def_static(labels_class, "load_buffer", [](torch::Tensor buffer, const std::string& array_backend) {
            check_array_backend(array_backend);
            return LabelsHolder::load_buffer(std::move(buffer));
        }, DOCSTRING,
        {torch::arg("buffer"), torch::arg("array_backend") = "torch"}
    );

    def_static(labels_class, "unsafe_from_ptr", [](int64_t ptr, const std::string& array_backend) -> Labels {
            check_array_backend(array_backend);
            auto* labels_ptr = reinterpret_cast<mts_labels_t*>(ptr);
            auto labels = metatensor::Labels::unsafe_from_ptr(labels_ptr);
            return LabelsHolder::from_metatensor(std::move(labels));
        }, DOCSTRING,
        {torch::arg("ptr"), torch::arg("array_backend") = "torch"}
    );


    auto block_class = m.class_<TensorBlockHolder>("TensorBlock");
    block_class
        .def(
            torch::init<torch::Tensor, Labels, std::vector<Labels>, Labels>(), DOCSTRING,
            {torch::arg("values"), torch::arg("samples"), torch::arg("components"), torch::arg("properties")}
        )
        .def("__repr__", &TensorBlockHolder::repr)
        .def("__str__", &TensorBlockHolder::repr)
        .def("__len__", &TensorBlockHolder::len )
        .def("copy", &TensorBlockHolder::copy, DOCSTRING, {torch::arg("deep") = true})
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
        .def_property("is_view", &TensorBlockHolder::is_view)
        .def("to", [](
                const TensorBlock& self,
                torch::IValue positional_1,
                torch::IValue positional_2,
                torch::optional<torch::Dtype> dtype,
                torch::optional<torch::Device> device,
                torch::optional<std::string> array_backend,
                bool non_blocking,
                torch::optional<std::string> arrays
            ) {
                return self->to_positional(
                    std::move(positional_1),
                    std::move(positional_2),
                    dtype,
                    device,
                    to_array_backend("`TensorBlock.to`", std::move(array_backend), std::move(arrays)),
                    non_blocking
                );
            }, DOCSTRING, {
            torch::arg("_0") = torch::IValue(),
            torch::arg("_1") = torch::IValue(),
            torch::arg("dtype") = torch::nullopt,
            torch::arg("device") = torch::nullopt,
            torch::arg("array_backend") = torch::nullopt,
            torch::arg("non_blocking") = false,
            torch::arg("arrays") = torch::nullopt
        })
        .def("save", &TensorBlockHolder::save, DOCSTRING, {torch::arg("file")})
        .def("save_buffer", &TensorBlockHolder::save_buffer)
        .def("release", [](TensorBlock self) -> int64_t {
            const auto* ptr = self->release().release();
            return reinterpret_cast<int64_t>(ptr);
        })
        .def("as_mts_block_t", [](const TensorBlock& self) -> int64_t {
            const auto* ptr = self->as_metatensor().as_mts_block_t();
            return reinterpret_cast<int64_t>(ptr);
        })
        .def_pickle(
            // __getstate__
            [](const TensorBlock& self){ return self->save_buffer(); },
            // __setstate__
            [](torch::Tensor buffer){ return metatensor_torch::load_block_buffer(buffer); }
        );

    def_static(block_class, "load", [](const std::string& file, const std::string& array_backend) {
            check_array_backend(array_backend);
            return TensorBlockHolder::load(file);
        }, DOCSTRING,
        {torch::arg("file"), torch::arg("array_backend") = "torch"}
    );

    def_static(block_class, "load_buffer", [](torch::Tensor buffer, const std::string& array_backend) {
            check_array_backend(array_backend);
            return TensorBlockHolder::load_buffer(std::move(buffer));
        }, DOCSTRING,
        {torch::arg("buffer"), torch::arg("array_backend") = "torch"}
    );

    def_static(block_class, "unsafe_from_ptr", [](int64_t ptr, const std::string& array_backend) -> TensorBlock {
            check_array_backend(array_backend);
            auto* block_ptr = reinterpret_cast<mts_block_t*>(ptr);
            auto block = metatensor::TensorBlock::unsafe_from_ptr(block_ptr);
            return TensorBlockHolder::from_metatensor(std::move(block));
        }, DOCSTRING,
        {torch::arg("ptr"), torch::arg("array_backend") = "torch"}
    );

    def_static(block_class, "unsafe_view_from_ptr", [](int64_t ptr, torch::IValue parent, const std::string& array_backend) -> TensorBlock {
            check_array_backend(array_backend);
            auto* block_ptr = reinterpret_cast<mts_block_t*>(ptr);
            auto block = metatensor::TensorBlock::unsafe_view_from_ptr(block_ptr);
            return TensorBlockHolder::view_from_metatensor(std::move(block), std::move(parent));
        }, DOCSTRING,
        {torch::arg("ptr"), torch::arg("parent"), torch::arg("array_backend") = "torch"}
    );

    auto tensor_class = m.class_<TensorMapHolder>("TensorMap");
    tensor_class
        .def(
            torch::init<Labels, std::vector<TensorBlock>>(), DOCSTRING,
            {torch::arg("keys"), torch::arg("blocks")}
        )
        .def("__len__", [](const TensorMap& self){ return self->keys()->count(); })
        .def("__repr__", [](const TensorMap& self){ return self->print(-1); })
        .def("__str__", [](const TensorMap& self){ return self->print(7); })
        .def("__getitem__", &TensorMapHolder::block_torch, DOCSTRING,
            {torch::arg("selection")}
        )
        .def("copy", &TensorMapHolder::copy, DOCSTRING, {torch::arg("deep") = true})
        .def("save", &TensorMapHolder::save, DOCSTRING, {torch::arg("file")})
        .def("save_buffer", &TensorMapHolder::save_buffer)
        .def("release", [](TensorMap self) -> int64_t {
            const auto* ptr = self->release().release();
            return reinterpret_cast<int64_t>(ptr);
        })
        .def("as_mts_tensormap_t", [](const TensorMap& self) -> int64_t {
            const auto* ptr = self->as_metatensor().as_mts_tensormap_t();
            return reinterpret_cast<int64_t>(ptr);
        })
        .def("items", &TensorMapHolder::items)
        .def_property("keys", &TensorMapHolder::keys)
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
            {torch::arg("keys_to_move"), torch::arg("fill_value") = 0, torch::arg("sort_samples") = true}
        )
        .def("keys_to_properties", &TensorMapHolder::keys_to_properties, DOCSTRING,
            {torch::arg("keys_to_move"), torch::arg("fill_value") = 0, torch::arg("sort_samples") = true}
        )
        .def("components_to_properties", &TensorMapHolder::components_to_properties, DOCSTRING,
            {torch::arg("dimensions")}
        )
        .def("set_info", &TensorMapHolder::set_info, DOCSTRING,
            {torch::arg("key"), torch::arg("value")}
        )
        .def("get_info", &TensorMapHolder::get_info, DOCSTRING,
            {torch::arg("key")}
        )
        .def("info", &TensorMapHolder::info)
        .def_property("sample_names", &TensorMapHolder::sample_names)
        .def_property("component_names", &TensorMapHolder::component_names)
        .def_property("property_names", &TensorMapHolder::property_names)
        .def_property("device", &TensorMapHolder::device)
        .def_property("dtype", &TensorMapHolder::scalar_type)
        .def_property("is_view", &TensorMapHolder::is_view)
        .def("to", [](
                const TensorMap& self,
                torch::IValue positional_1,
                torch::IValue positional_2,
                torch::optional<torch::Dtype> dtype,
                torch::optional<torch::Device> device,
                torch::optional<std::string> array_backend,
                bool non_blocking,
                torch::optional<std::string> arrays
            ) {
                return self->to_positional(
                    std::move(positional_1),
                    std::move(positional_2),
                    dtype,
                    device,
                    to_array_backend("`TensorMap.to`", std::move(array_backend), std::move(arrays)),
                    non_blocking
                );
            }, DOCSTRING, {
            torch::arg("_0") = torch::IValue(),
            torch::arg("_1") = torch::IValue(),
            torch::arg("dtype") = torch::nullopt,
            torch::arg("device") = torch::nullopt,
            torch::arg("array_backend") = torch::nullopt,
            torch::arg("non_blocking") = false,
            torch::arg("arrays") = torch::nullopt
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

    def_static(tensor_class, "load", [](const std::string& file, const std::string& array_backend) {
            check_array_backend(array_backend);
            return TensorMapHolder::load(file);
        }, DOCSTRING,
        {torch::arg("file"), torch::arg("array_backend") = "torch"}
    );

    def_static(tensor_class, "load_buffer", [](torch::Tensor buffer, const std::string& array_backend) {
            check_array_backend(array_backend);
            return TensorMapHolder::load_buffer(std::move(buffer));
        }, DOCSTRING,
        {torch::arg("buffer"), torch::arg("array_backend") = "torch"}
    );

    def_static(tensor_class, "unsafe_from_ptr", [](int64_t ptr, const std::string& array_backend) -> TensorMap {
            check_array_backend(array_backend);
            auto* tensor_ptr = reinterpret_cast<mts_tensormap_t*>(ptr);
            auto block = metatensor::TensorMap::unsafe_from_ptr(tensor_ptr);
            return TensorMapHolder::from_metatensor(std::move(block));
        }, DOCSTRING,
        {torch::arg("ptr"), torch::arg("array_backend") = "torch"}
    );

    def_static(tensor_class, "unsafe_view_from_ptr", [](int64_t ptr, torch::IValue parent, const std::string& array_backend) -> TensorMap {
            check_array_backend(array_backend);
            auto* tensor_ptr = reinterpret_cast<mts_tensormap_t*>(ptr);
            auto block = metatensor::TensorMap::unsafe_view_from_ptr(tensor_ptr);
            return TensorMapHolder::view_from_metatensor(std::move(block), std::move(parent));
        }, DOCSTRING,
        {torch::arg("ptr"), torch::arg("parent"), torch::arg("array_backend") = "torch"}
    );


    // standalone functions
    m.def("version() -> str", metatensor_torch::version);
    m.def("dtype_name(ScalarType dtype) -> str", scalar_type_name);

    m.def(
        "load(str file, *, str array_backend=\"torch\") -> __torch__.torch.classes.metatensor.TensorMap",
        [](const std::string& file, const std::string& array_backend) {
            check_array_backend(array_backend);
            return metatensor_torch::load(file);
        }
    );
    m.def(
        "load_buffer(Tensor buffer, *, str array_backend=\"torch\") -> __torch__.torch.classes.metatensor.TensorMap",
        [](torch::Tensor buffer, const std::string& array_backend) {
            check_array_backend(array_backend);
            return metatensor_torch::load_buffer(std::move(buffer));
        }
    );

    m.def(
        "load_block(str file, *, str array_backend=\"torch\") -> __torch__.torch.classes.metatensor.TensorBlock",
        [](const std::string& file, const std::string& array_backend) {
            check_array_backend(array_backend);
            return metatensor_torch::load_block(file);
        }
    );
    m.def(
        "load_block_buffer(Tensor buffer, *, str array_backend=\"torch\") -> __torch__.torch.classes.metatensor.TensorBlock",
        [](torch::Tensor buffer, const std::string& array_backend) {
            check_array_backend(array_backend);
            return metatensor_torch::load_block_buffer(std::move(buffer));
        }
    );

    m.def(
        "load_labels(str file, *, str array_backend=\"torch\") -> __torch__.torch.classes.metatensor.Labels",
        [](const std::string& file, const std::string& array_backend) {
            check_array_backend(array_backend);
            return metatensor_torch::load_labels(file);
        }
    );
    m.def(
        "load_labels_buffer(Tensor buffer, *, str array_backend=\"torch\") -> __torch__.torch.classes.metatensor.Labels",
        [](torch::Tensor buffer, const std::string& array_backend) {
            check_array_backend(array_backend);
            return metatensor_torch::load_labels_buffer(std::move(buffer));
        }
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
