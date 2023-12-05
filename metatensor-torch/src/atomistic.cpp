#include <cctype>
#include <cstring>

#include <algorithm>
#include <sstream>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <metatensor.hpp>

#include "metatensor/torch/atomistic.hpp"
#include "internal/scalar_type_name.hpp"

using namespace metatensor_torch;

NeighborsListOptionsHolder::NeighborsListOptionsHolder(
    double model_cutoff,
    bool full_list,
    std::string requestor
):
    model_cutoff_(model_cutoff),
    engine_cutoff_(model_cutoff),
    full_list_(full_list)
{
    this->add_requestor(std::move(requestor));
}


void NeighborsListOptionsHolder::add_requestor(std::string requestor) {
    if (requestor.empty()) {
        return;
    }

    for (const auto& existing: requestors_) {
        if (requestor == existing) {
            return;
        }
    }

    requestors_.emplace_back(requestor);
}

std::string NeighborsListOptionsHolder::repr() const {
    auto ss = std::ostringstream();

    ss << "NeighborsListOptions\n";
    ss << "    model_cutoff: " << std::to_string(model_cutoff_) << "\n";
    ss << "    full_list: " << (full_list_ ? "True" : "False") << "\n";

    if (!requestors_.empty()) {
        ss << "    requested by:\n";
        for (const auto& requestor: requestors_) {
            ss << "        - " << requestor << "\n";
        }
    }

    return ss.str();
}

std::string NeighborsListOptionsHolder::str() const {
    return "NeighborsListOptions(cutoff=" + std::to_string(model_cutoff_) + \
        ", full_list=" + (full_list_ ? "True" : "False") + ")";
}

std::string NeighborsListOptionsHolder::to_json() const {
    nlohmann::json result;

    result["type"] = "NeighborsListOptions";

    // Store cutoff using it's binary representation to ensure perfect
    // round-tripping of the data
    static_assert(sizeof(double) == sizeof(int64_t));
    int64_t int_cutoff = 0;
    std::memcpy(&int_cutoff, &this->model_cutoff_, sizeof(double));
    result["model_cutoff"] = int_cutoff;

    result["full_list"] = this->full_list_;

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

NeighborsListOptions NeighborsListOptionsHolder::from_json(const std::string& json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for NeighborsListOptions, expected an object");
    }

    if (!data.contains("type") || !data["type"].is_string()) {
        throw std::runtime_error("expected 'type' in JSON for NeighborsListOptions, did not find it");
    }

    if (data["type"] != "NeighborsListOptions") {
        throw std::runtime_error("'type' in JSON for NeighborsListOptions must be 'NeighborsListOptions'");
    }


    if (!data.contains("model_cutoff") || !data["model_cutoff"].is_number_integer()) {
        throw std::runtime_error("'model_cutoff' in JSON for NeighborsListOptions must be a number");
    }
    auto int_cutoff = data["model_cutoff"].get<int64_t>();

    if (!data.contains("full_list") || !data["full_list"].is_boolean()) {
        throw std::runtime_error("'full_list' in JSON for NeighborsListOptions must be a boolean");
    }
    auto full_list = data["full_list"].get<bool>();
    double model_cutoff = 0;
    std::memcpy(&model_cutoff, &int_cutoff, sizeof(double));

    return torch::make_intrusive<NeighborsListOptionsHolder>(model_cutoff, full_list);
}

// ========================================================================== //

static bool is_floating_point(torch::Dtype dtype) {
    return dtype == torch::kF16 || dtype == torch::kF32 || dtype == torch::kF64;
}

SystemHolder::SystemHolder(torch::Tensor species, torch::Tensor positions, torch::Tensor cell):
    species_(std::move(species)),
    positions_(std::move(positions)),
    cell_(std::move(cell))
{
    if (positions_.device() != species_.device() || cell_.device() != species_.device()) {
        C10_THROW_ERROR(ValueError,
            "`species`, `positions`, and `cell` must be on the same device, got " +
            species_.device().str() + ", " + positions_.device().str() + ", and " +
            cell_.device().str()
        );
    }

    if (species_.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError,
            "`species` must be a 1 dimensional tensor, got a tensor with " +
            std::to_string(species_.sizes().size()) + " dimensions"
        );
    }

    if (torch::canCast(species_.scalar_type(), torch::kInt32)) {
        species_ = species_.to(torch::kInt32);
    } else {
        C10_THROW_ERROR(ValueError,
            "`species` must be a tensor of integers, got " +
            scalar_type_name(species_.scalar_type()) + " instead"
        );
    }

    auto n_atoms = species_.size(0);
    if (positions_.sizes().size() != 2) {
        C10_THROW_ERROR(ValueError,
            "`positions` must be a 2 dimensional tensor, got a tensor with " +
            std::to_string(positions_.sizes().size()) + " dimensions"
        );
    }

    if (positions_.size(0) != n_atoms || positions_.size(1) != 3) {
        C10_THROW_ERROR(ValueError,
            "`positions` must be a (n_atoms x 3) tensor, got a tensor with shape [" +
            std::to_string(positions_.size(0)) + ", " + std::to_string(positions_.size(1)) + "]"
        );
    }

    if (!is_floating_point(positions_.scalar_type())) {
        C10_THROW_ERROR(ValueError,
            "`positions` must be a tensor of floating point data, got " +
            scalar_type_name(positions_.scalar_type()) + " instead"
        );
    }

    if (cell_.sizes().size() != 2) {
        C10_THROW_ERROR(ValueError,
            "`cell` must be a 2 dimensional tensor, got a tensor with " +
            std::to_string(cell_.sizes().size()) + " dimensions"
        );
    }

    if (cell_.size(0) != 3 || cell_.size(1) != 3) {
        C10_THROW_ERROR(ValueError,
            "`cell` must be a (3 x 3) tensor, got a tensor with shape [" +
            std::to_string(cell_.size(0)) + ", " + std::to_string(cell_.size(1)) + "]"
        );
    }

    if (cell_.scalar_type() != positions_.scalar_type()) {
        C10_THROW_ERROR(ValueError,
            "`cell` must be have the same dtype as `positions`, got " +
            scalar_type_name(cell_.scalar_type()) + " and " +
            scalar_type_name(positions_.scalar_type())
        );
    }
}


void SystemHolder::set_species(torch::Tensor species) {
    if (species.device() != this->device()) {
        C10_THROW_ERROR(ValueError,
            "new `species` must be on the same device as existing data, got " +
            species.device().str() + " and " + this->device().str()
        );
    }

    if (species.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError,
            "new `species` must be a 1 dimensional tensor, got a tensor with " +
            std::to_string(species.sizes().size()) + " dimensions"
        );
    }

    if (species.size(0) != this->size()) {
        C10_THROW_ERROR(ValueError,
            "new `species` must contain " + std::to_string(this->size()) + " entries, "
            "got a tensor with " + std::to_string(species.size(0)) + " values"
        );
    }

    if (torch::canCast(species.scalar_type(), torch::kInt32)) {
        species = species.to(torch::kInt32);
    } else {
        C10_THROW_ERROR(ValueError,
            "new `species` must be a tensor of integers, got " +
            scalar_type_name(species.scalar_type()) + " instead"
        );
    }

    this->species_ = std::move(species);
}


void SystemHolder::set_positions(torch::Tensor positions) {
    if (positions.device() != this->device()) {
        C10_THROW_ERROR(ValueError,
            "new `positions` must be on the same device as existing data, got " +
            positions.device().str() + " and " + this->device().str()
        );
    }

    if (positions.scalar_type() != this->scalar_type()) {
        C10_THROW_ERROR(ValueError,
            "new `positions` must have the same dtype as existing data, got " +
            scalar_type_name(positions.scalar_type()) + " and " + scalar_type_name(this->scalar_type())
        );
    }

    if (positions.sizes().size() != 2) {
        C10_THROW_ERROR(ValueError,
            "new `positions` must be a 2 dimensional tensor, got a tensor with " +
            std::to_string(positions.sizes().size()) + " dimensions"
        );
    }

    if (positions.size(0) != this->size() || positions.size(1) != 3) {
        C10_THROW_ERROR(ValueError,
            "new `positions` must be a (n_atoms x 3) tensor, got a tensor with shape [" +
            std::to_string(positions.size(0)) + ", " + std::to_string(positions.size(1)) + "]"
        );
    }

    this->positions_ = std::move(positions);
}


void SystemHolder::set_cell(torch::Tensor cell) {
    if (cell.device() != this->device()) {
        C10_THROW_ERROR(ValueError,
            "new `cell` must be on the same device as existing data, got " +
            cell.device().str() + " and " + this->device().str()
        );
    }

    if (cell.scalar_type() != this->scalar_type()) {
        C10_THROW_ERROR(ValueError,
            "new `cell` must have the same dtype as existing data, got " +
            scalar_type_name(cell.scalar_type()) + " and " + scalar_type_name(this->scalar_type())
        );
    }

    if (cell.sizes().size() != 2) {
        C10_THROW_ERROR(ValueError,
            "new `cell` must be a 2 dimensional tensor, got a tensor with " +
            std::to_string(cell.sizes().size()) + " dimensions"
        );
    }

    if (cell.size(0) != 3 || cell.size(1) != 3) {
        C10_THROW_ERROR(ValueError,
            "new `cell` must be a (3 x 3) tensor, got a tensor with shape [" +
            std::to_string(cell.size(0)) + ", " + std::to_string(cell.size(1)) + "]"
        );
    }

    this->cell_ = std::move(cell);
}


void SystemHolder::add_neighbors_list(NeighborsListOptions options, TorchTensorBlock neighbors) {
    // check the structure of the NL
    auto samples_names = neighbors->samples()->names();
    if (samples_names.size() != 5 ||
        samples_names[0] != "first_atom" ||
        samples_names[1] != "second_atom" ||
        samples_names[2] != "cell_shift_a" ||
        samples_names[3] != "cell_shift_b" ||
        samples_names[4] != "cell_shift_c"
    ) {
        C10_THROW_ERROR(ValueError,
            "invalid samples for `neighbors`: the samples names must be "
            "'first_atom', 'second_atom', 'cell_shift_a', 'cell_shift_b', 'cell_shift_c'"
        );
    }

    // TODO: we could check that the values of first_atom and second_atom match
    // entried in positions, but this might be a bit costly

    auto components = neighbors->components();
    if (components.size() != 1 || *components[0] != metatensor::Labels({"xyz"}, {{0}, {1}, {2}})) {
        C10_THROW_ERROR(ValueError,
            "invalid components for `neighbors`: there should be a single 'xyz'=[0, 1, 2] component"
        );
    }

    if (*neighbors->properties() != metatensor::Labels({"distance"}, {{0}})) {
        C10_THROW_ERROR(ValueError,
            "invalid properties for `neighbors`: there should be a single 'distance'=0 property"
        );
    }

    if (!neighbors->gradients_list().empty()) {
        C10_THROW_ERROR(ValueError, "`neighbors` should not have any gradients");
    }

    const auto& values = neighbors->values();
    if (values.device() != this->device()) {
        C10_THROW_ERROR(ValueError,
            "`neighbors` device (" + values.device().str() + ") does not match "
            "this system's device (" + this->device().str() +")"
        );
    }

    if (values.scalar_type() != this->scalar_type()) {
        C10_THROW_ERROR(ValueError,
            "`neighbors` dtype (" + scalar_type_name(neighbors->values().scalar_type()) +
            ") does not match this system's dtype (" + scalar_type_name(this->scalar_type()) +")"
        );
    }

    // actually add the neighbors list
    auto it = neighbors_.find(options);
    if (it != neighbors_.end()) {
        C10_THROW_ERROR(ValueError,
            "the neighbors list for " + options->str() + " already exists in this system"
        );
    }

    neighbors_.emplace(std::move(options), std::move(neighbors));
}

TorchTensorBlock SystemHolder::get_neighbors_list(NeighborsListOptions options) const {
    auto it = neighbors_.find(options);
    if (it == neighbors_.end()) {
        C10_THROW_ERROR(ValueError,
            "No neighbors list for " + options->str() + " was found.\n"
            "Is it part of the `requested_neighbors_lists` for this model?"
        );
    }
    return it->second;
}

std::vector<NeighborsListOptions> SystemHolder::known_neighbors_lists() const {
    auto result = std::vector<NeighborsListOptions>();
    for (const auto& it: neighbors_) {
        result.emplace_back(it.first);
    }
    return result;
}

bool is_alpha_or_digit(char c) {
    return (('0' <= c) && (c <= '9')) ||
           (('a' <= c) && (c <= 'z')) ||
           (('A' <= c) && (c <= 'Z'));
}

static bool valid_ident(const std::string& string) {
    if (string.empty()) {
        return false;
    }

    for (const auto c: string) {
        if (!(is_alpha_or_digit(c) || c == '_' || c == '-')) {
            return false;
        }
    }
    return true;
}

static std::string string_lower(const std::string& value) {
    auto copy = value;
    std::transform(copy.begin(), copy.end(), copy.begin(),
        [](unsigned char c){ return std::tolower(c); }
    );
    return copy;
}

static auto INVALID_DATA_NAMES = std::unordered_set<std::string>{
    "species",
    "positions", "position",
    "cell",
    "neighbors", "neighbor"
};

void SystemHolder::add_data(std::string name, TorchTensorBlock values) {
    if (!valid_ident(name)) {
        C10_THROW_ERROR(ValueError,
            "custom data name '" + name + "' is invalid: only [a-z A-Z 0-9 _-] are accepted"
        );
    }

    if (INVALID_DATA_NAMES.find(string_lower(name)) != INVALID_DATA_NAMES.end()) {
        C10_THROW_ERROR(ValueError,
            "custom data can not be named '" + name + "'"
        );
    }

    if (data_.find(name) != data_.end()) {
        C10_THROW_ERROR(ValueError,
            "custom data '" + name + "' is already present in this system"
        );
    }

    const auto& values_tensor = values->values();
    if (values_tensor.device() != this->device()) {
        C10_THROW_ERROR(ValueError,
            "device (" + values_tensor.device().str() + ") of the custom data "
            "'" + name + "' does not match this system device (" + this->device().str() +")"
        );
    }

    if (values_tensor.scalar_type() != this->scalar_type()) {
        C10_THROW_ERROR(ValueError,
            "dtype (" + scalar_type_name(values_tensor.scalar_type()) + ") of " +
            "custom data '" + name + "' does not match this system " +
            "dtype (" + scalar_type_name(this->scalar_type()) +")"
        );
    }

    data_.emplace(std::move(name), std::move(values));
}

TorchTensorBlock SystemHolder::get_data(std::string name) const {
    if (INVALID_DATA_NAMES.find(string_lower(name)) != INVALID_DATA_NAMES.end()) {
        C10_THROW_ERROR(ValueError,
            "custom data can not be named '" + name + "'"
        );
    }

    auto it = data_.find(name);
    if (it == data_.end()) {
        C10_THROW_ERROR(ValueError,
            "no data for '" + name + "' found in this system"
        );
    }

    TORCH_WARN_ONCE(
        "custom data '", name, "' is experimental, please contact metatensor's ",
        "developers to add this data as a member of the `System` class"
    );
    return it->second;
}


std::vector<std::string> SystemHolder::known_data() const {
    auto result = std::vector<std::string>();
    for (const auto& it: data_) {
        result.emplace_back(it.first);
    }
    return result;
}


// ========================================================================== //

static nlohmann::json model_output_to_json(const ModelOutputHolder& self) {
    nlohmann::json result;

    result["type"] = "ModelOutput";
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

    if (!data.contains("type") || !data["type"].is_string()) {
        throw std::runtime_error("expected 'type' in JSON for ModelOutput, did not find it");
    }

    if (data["type"] != "ModelOutput") {
        throw std::runtime_error("'type' in JSON for ModelOutput must be 'ModelOutput'");
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

    result["type"] = "ModelCapabilities";
    result["length_unit"] = this->length_unit;
    result["species"] = this->species;

    auto outputs = nlohmann::json::object();
    for (const auto& it: this->outputs) {
        outputs[it.key()] = model_output_to_json(*it.value());
    }
    result["outputs"] = outputs;

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

ModelCapabilities ModelCapabilitiesHolder::from_json(const std::string& json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelCapabilities, expected an object");
    }

    if (!data.contains("type") || !data["type"].is_string()) {
        throw std::runtime_error("expected 'type' in JSON for ModelCapabilities, did not find it");
    }

    if (data["type"] != "ModelCapabilities") {
        throw std::runtime_error("'type' in JSON for ModelCapabilities must be 'ModelCapabilities'");
    }

    auto result = torch::make_intrusive<ModelCapabilitiesHolder>();
    if (data.contains("length_unit")) {
        if (!data["length_unit"].is_string()) {
            throw std::runtime_error("'length_unit' in JSON for ModelCapabilities must be a string");
        }
        result->length_unit = data["length_unit"];
    }

    if (data.contains("species")) {
        if (!data["species"].is_array()) {
            throw std::runtime_error("'species' in JSON for ModelCapabilities must be an array");
        }

        for (const auto& species: data["species"]) {
            if (!species.is_number_integer()) {
                throw std::runtime_error("'species' in JSON for ModelCapabilities must be an array of integers");
            }
            result->species.emplace_back(species);
        }
    }

    if (data.contains("outputs")) {
        if (!data["outputs"].is_object()) {
            throw std::runtime_error("'outputs' in JSON for ModelCapabilities must be an object");
        }

        for (const auto& output: data["outputs"].items()) {
            result->outputs.insert(output.key(), model_output_from_json(output.value()));
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

    result["type"] = "ModelEvaluationOptions";
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

    if (!data.contains("type") || !data["type"].is_string()) {
        throw std::runtime_error("expected 'type' in JSON for ModelEvaluationOptions, did not find it");
    }

    if (data["type"] != "ModelEvaluationOptions") {
        throw std::runtime_error("'type' in JSON for ModelEvaluationOptions must be 'ModelEvaluationOptions'");
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


/******************************************************************************/

#include "internal/shared_libraries.hpp"

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
    auto current_mts_version = Version(mts_version());

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
