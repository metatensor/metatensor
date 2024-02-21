#include <cstring>

#include <array>
#include <sstream>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <metatensor.hpp>

#include "metatensor/torch/atomistic/model.hpp"
#include "metatensor/torch/misc.hpp"

#include "../internal/scalar_type_name.hpp"
#include "../internal/shared_libraries.hpp"

using namespace metatensor_torch;

static nlohmann::json model_output_to_json(const ModelOutputHolder& self) {
    nlohmann::json result;

    result["class"] = "ModelOutput";
    result["quantity"] = self.quantity;
    result["unit"] = self.unit;
    result["per_atom"] = self.per_atom;
    result["explicit_gradients"] = self.explicit_gradients;

    return result;
}

std::string ModelOutputHolder::to_json() const {
    return model_output_to_json(*this).dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

static ModelOutput model_output_from_json(const nlohmann::json& data) {
    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelOutput, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for ModelOutput, did not find it");
    }

    if (data["class"] != "ModelOutput") {
        throw std::runtime_error("'class' in JSON for ModelOutput must be 'ModelOutput'");
    }

    auto result = torch::make_intrusive<ModelOutputHolder>();
    if (data.contains("quantity")) {
        if (!data["quantity"].is_string()) {
            throw std::runtime_error("'quantity' in JSON for ModelOutput must be a string");
        }
        result->quantity = data["quantity"];
    }

    if (data.contains("unit")) {
        if (!data["unit"].is_string()) {
            throw std::runtime_error("'unit' in JSON for ModelOutput must be a string");
        }
        result->unit = data["unit"];
    }

    if (data.contains("per_atom")) {
        if (!data["per_atom"].is_boolean()) {
            throw std::runtime_error("'per_atom' in JSON for ModelOutput must be a boolean");
        }
        result->per_atom = data["per_atom"];
    }

    if (data.contains("explicit_gradients")) {
        if (!data["explicit_gradients"].is_array()) {
            throw std::runtime_error("'explicit_gradients' in JSON for ModelOutput must be an array");
        }

        for (const auto& gradient: data["explicit_gradients"]) {
            if (!gradient.is_string()) {
                throw std::runtime_error("'explicit_gradients' in JSON for ModelOutput must be an array of strings");
            }
            result->explicit_gradients.emplace_back(gradient);
        }
    }

    return result;
}

ModelOutput ModelOutputHolder::from_json(const std::string& json) {
    auto data = nlohmann::json::parse(json);
    return model_output_from_json(data);
}


std::string ModelCapabilitiesHolder::to_json() const {
    nlohmann::json result;

    result["class"] = "ModelCapabilities";

    auto outputs = nlohmann::json::object();
    for (const auto& it: this->outputs) {
        outputs[it.key()] = model_output_to_json(*it.value());
    }
    result["outputs"] = outputs;
    result["atomic_types"] = this->atomic_types;

    // Store interaction_range using it's binary representation to ensure
    // perfect round-tripping of the data
    static_assert(sizeof(double) == sizeof(int64_t));
    int64_t int_interaction_range = 0;
    std::memcpy(&int_interaction_range, &this->interaction_range, sizeof(double));
    result["interaction_range"] = int_interaction_range;

    result["length_unit"] = this->length_unit;
    result["supported_devices"] = this->supported_devices;

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

ModelCapabilities ModelCapabilitiesHolder::from_json(const std::string& json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelCapabilities, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for ModelCapabilities, did not find it");
    }

    if (data["class"] != "ModelCapabilities") {
        throw std::runtime_error("'class' in JSON for ModelCapabilities must be 'ModelCapabilities'");
    }

    auto result = torch::make_intrusive<ModelCapabilitiesHolder>();
    if (data.contains("outputs")) {
        if (!data["outputs"].is_object()) {
            throw std::runtime_error("'outputs' in JSON for ModelCapabilities must be an object");
        }

        for (const auto& output: data["outputs"].items()) {
            result->outputs.insert(output.key(), model_output_from_json(output.value()));
        }
    }

    if (data.contains("atomic_types")) {
        if (!data["atomic_types"].is_array()) {
            throw std::runtime_error("'atomic_types' in JSON for ModelCapabilities must be an array");
        }

        for (const auto& type: data["atomic_types"]) {
            if (!type.is_number_integer()) {
                throw std::runtime_error("'atomic_types' in JSON for ModelCapabilities must be an array of integers");
            }
            result->atomic_types.emplace_back(type);
        }
    }

    if (data.contains("interaction_range")) {
        if (!data["interaction_range"].is_number_integer()) {
            throw std::runtime_error("'interaction_range' in JSON for ModelCapabilities must be a number");
        }

        auto int_interaction_range = data["interaction_range"].get<int64_t>();
        double interaction_range = 0;
        std::memcpy(&interaction_range, &int_interaction_range, sizeof(double));

        result->interaction_range = interaction_range;
    }

    if (data.contains("length_unit")) {
        if (!data["length_unit"].is_string()) {
            throw std::runtime_error("'length_unit' in JSON for ModelCapabilities must be a string");
        }
        result->length_unit = data["length_unit"];
    }

    if (data.contains("supported_devices")) {
        if (!data["supported_devices"].is_array()) {
            throw std::runtime_error("'supported_devices' in JSON for ModelCapabilities must be an array");
        }

        for (const auto& device: data["supported_devices"]) {
            if (!device.is_string()) {
                throw std::runtime_error("'supported_devices' in JSON for ModelCapabilities must be an array of string");
            }
            result->supported_devices.emplace_back(device);
        }
    }

    return result;
}


static void check_selected_atoms(const torch::optional<TorchLabels>& selected_atoms) {
    if (selected_atoms) {
        if (selected_atoms.value()->names() != std::vector<std::string>{"system", "atom"}) {
            std::ostringstream oss;
            oss << '[';
            for (const auto& name: selected_atoms.value()->names()) {
                oss << '\'' << name << "', ";
            }
            oss << ']';

            C10_THROW_ERROR(ValueError,
                "invalid `selected_atoms` names: expected ['system', 'atom'], "
                "got " + oss.str()
            );
        }
    }
}

ModelEvaluationOptionsHolder::ModelEvaluationOptionsHolder(
    std::string length_unit_,
    torch::Dict<std::string, ModelOutput> outputs_,
    torch::optional<TorchLabels> selected_atoms
):
    length_unit(std::move(length_unit_)),
    outputs(outputs_),
    selected_atoms_(std::move(selected_atoms))
{
    check_selected_atoms(selected_atoms_);
}


void ModelEvaluationOptionsHolder::set_selected_atoms(torch::optional<TorchLabels> selected_atoms) {
    check_selected_atoms(selected_atoms);
    selected_atoms_ = std::move(selected_atoms);
}


std::string ModelEvaluationOptionsHolder::to_json() const {
    nlohmann::json result;

    result["class"] = "ModelEvaluationOptions";
    result["length_unit"] = this->length_unit;

    if (this->selected_atoms_) {
        const auto& selected_atoms = this->selected_atoms_.value();

        auto selected_json = nlohmann::json::object();
        selected_json["names"] = selected_atoms->names();
        auto values = selected_atoms->values().to(torch::kCPU).contiguous();
        auto size = static_cast<size_t>(selected_atoms->size() * selected_atoms->count());
        selected_json["values"] = std::vector<int32_t>(
            values.data_ptr<int32_t>(),
            values.data_ptr<int32_t>() + size
        );

        result["selected_atoms"] = std::move(selected_json);
    } else {
        result["selected_atoms"] = nlohmann::json();
    }

    auto outputs = nlohmann::json::object();
    for (const auto& it: this->outputs) {
        outputs[it.key()] = model_output_to_json(*it.value());
    }
    result["outputs"] = outputs;

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

ModelEvaluationOptions ModelEvaluationOptionsHolder::from_json(const std::string& json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelEvaluationOptions, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for ModelEvaluationOptions, did not find it");
    }

    if (data["class"] != "ModelEvaluationOptions") {
        throw std::runtime_error("'class' in JSON for ModelEvaluationOptions must be 'ModelEvaluationOptions'");
    }

    auto result = torch::make_intrusive<ModelEvaluationOptionsHolder>();
    if (data.contains("length_unit")) {
        if (!data["length_unit"].is_string()) {
            throw std::runtime_error("'length_unit' in JSON for ModelEvaluationOptions must be a string");
        }
        result->length_unit = data["length_unit"];
    }

    if (data.contains("selected_atoms")) {
        if (data["selected_atoms"].is_null()) {
            // nothing to do
        } else {
            if (!data["selected_atoms"].is_object()) {
                throw std::runtime_error("'selected_atoms' in JSON for ModelEvaluationOptions must be an object");
            }

            if (!data["selected_atoms"].contains("names") || !data["selected_atoms"]["names"].is_array()) {
                throw std::runtime_error("'selected_atoms.names' in JSON for ModelEvaluationOptions must be an array");
            }

            auto names = std::vector<std::string>();
            for (const auto& name: data["selected_atoms"]["names"]) {
                if (!name.is_string()) {
                    throw std::runtime_error(
                        "'selected_atoms.names' in JSON for ModelEvaluationOptions must be an array of strings"
                    );
                }
                names.emplace_back(name.get<std::string>());
            }


            if (!data["selected_atoms"].contains("values") || !data["selected_atoms"]["values"].is_array()) {
                throw std::runtime_error("'selected_atoms.values' in JSON for ModelEvaluationOptions must be an array");
            }

            auto values = std::vector<int32_t>();
            for (const auto& value: data["selected_atoms"]["values"]) {
                if (!value.is_number_integer()) {
                    throw std::runtime_error(
                        "'selected_atoms.values' in JSON for ModelEvaluationOptions must be an array of integers"
                    );
                }
                values.emplace_back(value.get<int32_t>());
            }
            assert(values.size() % 2 == 0);

            result->set_selected_atoms(torch::make_intrusive<LabelsHolder>(
                std::move(names),
                torch::tensor(values).reshape({-1, 2})
            ));
        }
    }

    if (data.contains("outputs")) {
        if (!data["outputs"].is_object()) {
            throw std::runtime_error("'outputs' in JSON for ModelEvaluationOptions must be an object");
        }

        for (const auto& output: data["outputs"].items()) {
            result->outputs.insert(output.key(), model_output_from_json(output.value()));
        }
    }

    return result;
}

static std::string record_to_string(std::tuple<at::DataPtr, size_t> data) {
    return std::string(
        static_cast<char*>(std::get<0>(data).get()),
        std::get<1>(data)
    );
}

struct Version {
    Version(std::string version): string(std::move(version)) {
        size_t pos = 0;
        try {
            this->major = std::stoi(this->string, &pos);
        } catch (const std::invalid_argument&) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }

        if (this->string[pos] != '.' || this->string.size() == pos) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }

        auto minor_version = this->string.substr(pos + 1);
        try {
            this->minor = std::stoi(minor_version, &pos);
        } catch (const std::invalid_argument&) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }
    }

    /// Check if two version are compatible. `same_minor` indicates whether two
    /// versions should have the same major AND minor number to be considered
    /// compatible.
    bool is_compatible(const Version& other, bool same_minor = false) const {
        if (this->major != other.major) {
            return false;
        }

        if (this->major == 0) {
            same_minor = true;
        }

        if (same_minor && this->minor != other.minor) {
            return false;
        }

        return true;
    }

    std::string string;
    int major = 0;
    int minor = 0;
};

struct Extension {
    std::string name;
    std::string path;
};

void from_json(const nlohmann::json& json, Extension& extension) {
    json.at("name").get_to(extension.name);
    json.at("path").get_to(extension.path);
}

void metatensor_torch::check_atomistic_model(std::string path) {
    auto reader = caffe2::serialize::PyTorchStreamReader(path);

    if (!reader.hasRecord("extra/metatensor-version")) {
        C10_THROW_ERROR(ValueError,
            "file at '" + path + "' does not contain a metatensor atomistic model"
        );
    }

    auto recorded_mts_version = Version(record_to_string(
        reader.getRecord("extra/metatensor-version")
    ));
    auto current_mts_version = Version(metatensor_torch::version());

    if (!current_mts_version.is_compatible(recorded_mts_version)) {
        TORCH_WARN(
            "Current metatensor version (", current_mts_version.string, ") ",
            "is not compatible with the version (", recorded_mts_version.string,
            ") used to export the model at '", path, "'; proceed at your own risk."
        );
    }

    auto recorded_torch_version = Version(record_to_string(
        reader.getRecord("extra/torch-version")
    ));
    auto current_torch_version = Version(TORCH_VERSION);
    if (!current_torch_version.is_compatible(recorded_torch_version, true)) {
        TORCH_WARN(
            "Current torch version (", current_torch_version.string, ") ",
            "is not compatible with the version (", recorded_torch_version.string,
            ") used to export the model at '", path, "'; proceed at your own risk."
        );
    }

    // Check that the extensions loaded while the model was exported are also
    // loaded now. Since the model can be exported from a different machine, or
    // the extensions might change how they organize code, we only try to do
    // fuzzy matching on the file name, and warn if we can not find a match.
    std::vector<Extension> extensions = nlohmann::json::parse(record_to_string(
        reader.getRecord("extra/extensions")
    ));

    auto loaded_libraries = metatensor_torch::details::get_loaded_libraries();

    for (const auto& extension: extensions) {
        auto found = false;
        for (const auto& library: loaded_libraries) {
            if (library.find(extension.name) != std::string::npos) {
                found = true;
                break;
            }
        }

        if (!found) {
            TORCH_WARN(
                "The model at '", path, "' was exported with extension '",
                extension.name, "' loaded (from '", extension.path, "'), ",
                "but it does not seem to be currently loaded; proceed at your own risk."
            );
        }
    }
}

torch::jit::Module metatensor_torch::load_atomistic_model(
    std::string path,
    c10::optional<c10::Device> device
) {
    check_atomistic_model(path);
    return torch::jit::load(path, device);
}
