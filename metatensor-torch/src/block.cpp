#include <memory>
#include <string>

#include <metatensor.hpp>

#include "metatensor/torch/array.hpp"
#include "metatensor/torch/block.hpp"
#include "metatensor/torch/misc.hpp"

#include "internal/utils.hpp"

using namespace metatensor_torch;


static std::vector<metatensor::Labels> components_from_torch(const std::vector<Labels>& components) {
    auto result = std::vector<metatensor::Labels>();
    for (const auto& component: components) {
        result.push_back(component->as_metatensor());
    }
    return result;
}

TensorBlockHolder::TensorBlockHolder(
    torch::Tensor data,
    Labels samples,
    std::vector<Labels> components,
    Labels properties
):
    TensorBlockHolder(
        metatensor::TensorBlock(
            std::make_unique<TorchDataArray>(std::move(data)),
            samples->as_metatensor(),
            components_from_torch(components),
            properties->as_metatensor()
        ),
        /* parameter */ "",
        /* parent */ torch::IValue()
    )
{

    auto values_device = this->values().device();
    if (values_device != this->samples()->values().device()) {
        C10_THROW_ERROR(ValueError,
            "cannot create TensorBlock: values and samples must be on the same device, "
            "got " + values_device.str() + " and " + this->samples()->values().device().str()
        );
    }
    for (const auto& component : this->components()) {
        if (values_device != component->values().device()) {
            C10_THROW_ERROR(ValueError,
                "cannot create TensorBlock: values and components must be on the same device, "
                "got " + values_device.str() + " and " + component->values().device().str()
            );
        }
    }
    if (values_device != this->properties()->values().device()) {
        C10_THROW_ERROR(ValueError,
                "cannot create TensorBlock: values and properties must be on the same device, "
                "got " + values_device.str() + " and " + this->properties()->values().device().str()
        );
    }
}


TensorBlockHolder::TensorBlockHolder(metatensor::TensorBlock block, torch::IValue parent):
    TensorBlockHolder(std::move(block), "", std::move(parent))
{}

TensorBlockHolder::TensorBlockHolder(metatensor::TensorBlock block, std::string parameter, torch::IValue parent):
    block_(std::move(block)),
    parameter_(std::move(parameter)),
    parent_(std::move(parent))
{}

TensorBlock TensorBlockHolder::copy() const {
    return torch::make_intrusive<TensorBlockHolder>(this->block_.clone(), torch::IValue());
}

TensorBlock TensorBlockHolder::to(
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device
) const {
    auto values = this->values().to(
        dtype,
        /*layout*/ torch::nullopt,
        device,
        /*pin_memory*/ torch::nullopt,
        /*non_blocking*/ false,
        /*copy*/ false,
        /*memory_format*/ torch::MemoryFormat::Preserve
    );

    auto samples = this->samples()->to(device);
    auto components = std::vector<torch::intrusive_ptr<LabelsHolder>>();
    for (const auto& component : this->components()) {
        components.push_back(component->to(device));
    }
    auto properties = this->properties()->to(device);

    auto block = torch::make_intrusive<TensorBlockHolder>(values, samples, components, properties);
    for (const auto& parameter : this->gradients_list()) {
        auto gradient = TensorBlockHolder(
            this->block_.gradient(parameter),
            torch::IValue()
        );

        block->add_gradient(parameter, gradient.to(dtype, device));
    }
    return block;
}

TensorBlock TensorBlockHolder::to_positional(
    torch::IValue positional_1,
    torch::IValue positional_2,
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device,
    torch::optional<std::string> arrays
) const {
    if (arrays.value_or("torch") != "torch") {
        C10_THROW_ERROR(ValueError,
            "`arrays` must be None or 'torch', got '" + arrays.value() + "' instead"
        );
    }

    auto [parsed_dtype, parsed_device] = to_arguments_parse(
        positional_1,
        positional_2,
        dtype,
        device,
        "`TensorBlock.to`"
    );

    return this->to(parsed_dtype, parsed_device);
}

torch::Tensor TensorBlockHolder::values() const {
    // const_cast is fine here, because the returned torch::Tensor does not
    // allow modifications to the underlying mts_array (only to the values
    // inside the tensor).
    auto array = const_cast<metatensor::TensorBlock&>(block_).mts_array();

    mts_data_origin_t origin = 0;
    metatensor::details::check_status(array.origin(array.ptr, &origin));
    if (origin != TORCH_DATA_ORIGIN) {
        C10_THROW_ERROR(ValueError,
            "this TensorBlock does not contain a C++ torch Tensor"
        );
    }

    auto* ptr = static_cast<metatensor::DataArrayBase*>(array.ptr);
    auto* wrapper = dynamic_cast<TorchDataArray*>(ptr);
    if (wrapper == nullptr) {
        C10_THROW_ERROR(ValueError,
            "this TensorBlock does not contain a C++ torch Tensor"
        );
    }

    return wrapper->tensor();
}

void TensorBlockHolder::set_values(const torch::Tensor& new_values) {
        throw std::runtime_error(
            "Direct assignment to `values` is not possible. "
            "Please use block.values[:] = new_values instead."
        );
    }

Labels TensorBlockHolder::labels(uintptr_t axis) const {
    return torch::make_intrusive<LabelsHolder>(block_.labels(axis));
}


void TensorBlockHolder::add_gradient(const std::string& parameter, TensorBlock gradient) {
    // we need to move the tensor block in `add_gradient`, but we can not move
    // out of the `torch::intrusive_ptr` in `TensorBlock`. So we create a
    // new temporary block, increasing the reference count to the values and
    // metadata of gradient.
    auto gradient_block = metatensor::TensorBlock(
        std::make_unique<TorchDataArray>(gradient->values()),
        gradient->samples()->as_metatensor(),
        components_from_torch(gradient->components()),
        gradient->properties()->as_metatensor()
    );

    if (gradient->values().device() != this->values().device()) {
        C10_THROW_ERROR(ValueError,
            "values and the new gradient must be on the same device, "
            "got " + this->values().device().str() + " and " + gradient->values().device().str()
        );
    }
    if (gradient->values().scalar_type() != this->values().scalar_type()) {
        C10_THROW_ERROR(TypeError,
            "values and the new gradient must have the same dtype, "
            "got " + scalar_type_name(gradient->values().scalar_type()) +
            " and " + scalar_type_name(this->values().scalar_type())
        );
    }

    block_.add_gradient(parameter, std::move(gradient_block));
}

bool TensorBlockHolder::has_gradient(const std::string& parameter) const {
    auto list = this->block_.gradients_list();
    auto it = std::find(std::begin(list), std::end(list), parameter);
    return it != std::end(list);
}

TensorBlock TensorBlockHolder::gradient(TensorBlock self, const std::string& parameter) {
    // handle recursive gradients
    std::string gradient_parameter;
    if (!self->parameter_.empty()) {
        gradient_parameter = self->parameter_ + "/" + parameter;
    } else {
        gradient_parameter = parameter;
    }

    return torch::make_intrusive<TensorBlockHolder>(self->block_.gradient(parameter), gradient_parameter, self);
}

std::vector<std::tuple<std::string, TensorBlock>> TensorBlockHolder::gradients(TensorBlock self) {
    auto result = std::vector<std::tuple<std::string, TensorBlock>>();
    for (const auto& parameter: self->gradients_list()) {
        result.emplace_back(parameter, TensorBlockHolder::gradient(self, parameter));
    }
    return result;
}

static void print_labels(std::ostringstream& output, const metatensor::Labels& labels, const char* labels_kind) {
    output << "    " << labels_kind << " (" << labels.count() << "): ";
    output << "[";
    auto first = true;
    for (const auto& name: labels.names()) {
        if (!first) {
            output << ", ";
        }
        output << '\'' << name << '\'';
        first = false;
    }
    output << "]\n";
}

std::string TensorBlockHolder::repr() const {
    auto output = std::ostringstream();

    if (parameter_.empty()) {
        output << "TensorBlock\n";
    } else {
        output << "Gradient TensorBlock ('" << parameter_ << "')\n";
    }

    print_labels(output, block_.samples(), "samples");

    auto components = block_.components();
    output << "    components (";
    auto first = true;
    for (const auto& component: components) {
        if (!first) {
            output << ", ";
        }
        output << component.count();
        first = false;
    }

    output << "): [";
    first = true;
    for (const auto& component: components) {
        if (!first) {
            output << ", ";
        }
        assert(component.size() == 1);
        output << '\'' << component.names()[0] << '\'';
        first = false;
    }
    output << "]\n";

    print_labels(output, block_.properties(), "properties");

    auto gradients = block_.gradients_list();
    output << "    gradients: ";
    if (gradients.empty()) {
        output << "None\n";
    } else {
        first = true;
        output << "[";
        for (const auto& parameter: gradients) {
            if (!first) {
                output << ", ";
            }
            output << '\'' << parameter << '\'';
            first = false;
        }
        output << "]\n";
    }

    return output.str();
}

TensorBlock TensorBlockHolder::load(const std::string& path) {
    return torch::make_intrusive<TensorBlockHolder>(
        TensorBlockHolder(
            metatensor::io::load_block(path, details::create_torch_array),
            /*parent=*/torch::IValue()
        )
    );
}

TensorBlock TensorBlockHolder::load_buffer(torch::Tensor buffer) {
    if (buffer.scalar_type() != torch::kUInt8) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a tensor of uint8, not " +
            scalar_type_name(buffer.scalar_type())
        );
    }

    if (buffer.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a 1-dimensional tensor"
        );
    }

    auto block = metatensor::io::load_block_buffer(
        buffer.data_ptr<uint8_t>(),
        static_cast<size_t>(buffer.size(0)),
        details::create_torch_array
    );

    return torch::make_intrusive<TensorBlockHolder>(
        TensorBlockHolder(std::move(block), /*parent=*/torch::IValue())
    );
}


void TensorBlockHolder::save(const std::string& path) const {
    // check that device is CPU
    if (this->values().device() != torch::kCPU) {
        C10_THROW_ERROR(ValueError,
            "cannot save TensorBlock with device " + this->values().device().str() +
            ", only CPU is supported"
        );
    }
    // check that dtype is float64
    if (this->scalar_type() != torch::kFloat64) {
        C10_THROW_ERROR(ValueError,
            "cannot save TensorBlock with dtype " + scalar_type_name(this->scalar_type()) +
            ", only float64 is supported"
        );
    }
    return metatensor::io::save(path, this->as_metatensor());
}

torch::Tensor TensorBlockHolder::save_buffer() const {
    // check that device is CPU
    if (this->values().device() != torch::kCPU) {
        C10_THROW_ERROR(ValueError,
            "cannot save TensorBlock with device " + this->values().device().str() +
            ", only CPU is supported"
        );
    }
    // check that dtype is float64
    if (this->scalar_type() != torch::kFloat64) {
        C10_THROW_ERROR(ValueError,
            "cannot save TensorBlock with dtype " + scalar_type_name(this->scalar_type()) +
            ", only float64 is supported"
        );
    }
    auto buffer = metatensor::io::save_buffer(this->as_metatensor());
    // move the buffer to the heap so it can escape this function
    // `torch::from_blob` does not take ownership of the data,
    // so we need to register a custom deleter to clean up when
    // the tensor is no longer used
    auto* buffer_data = new std::vector<uint8_t>(std::move(buffer));

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
}
