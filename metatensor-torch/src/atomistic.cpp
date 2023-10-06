#include <cctype>
#include <cstring>
#include <sstream>

#include <torch/torch.h>

#include <metatensor.hpp>

#include "metatensor/torch/atomistic.hpp"

using namespace metatensor_torch;


std::string NeighborsListOptionsHolder::repr() const {
    return "NeighborsListOptions(model_cutoff=" + std::to_string(model_cutoff_) + \
        ", full_list=" + (full_list_ ? "True" : "False") + ")";
}

// ========================================================================== //

SystemHolder::SystemHolder(TorchTensorBlock positions_, TorchTensorBlock cell_):
    positions(std::move(positions_)),
    cell(std::move(cell_))
{
    // check the positions
    auto samples_names = positions->samples()->names();
    if (samples_names.size() != 2 || samples_names[0] != "atom" || samples_names[1] != "species") {
        C10_THROW_ERROR(ValueError,
            "invalid samples for `positions`: the samples names must be "
            "'atom' and 'species'"
        );
    }

    auto components = positions->components();
    if (components.size() != 1 || *components[0] != metatensor::Labels({"xyz"}, {{0}, {1}, {2}})) {
        C10_THROW_ERROR(ValueError,
            "invalid components for `positions`: there should be a single 'xyz'=[0, 1, 2] component"
        );
    }

    if (*positions->properties() != metatensor::Labels({"position"}, {{0}})) {
        C10_THROW_ERROR(ValueError,
            "invalid properties for `positions`: there should be a single 'positions'=0 property"
        );
    }

    if (!positions->gradients_list().empty()) {
        C10_THROW_ERROR(ValueError, "`positions` should not have any gradients");
    }

    // check the cell
    if (*cell->samples() != metatensor::Labels({"_"}, {{0}})) {
        C10_THROW_ERROR(ValueError,
            "invalid samples for `cell`: there should be a single '_'=0 sample"
        );
    }

    components = cell->components();
    if (components.size() != 2) {
        C10_THROW_ERROR(ValueError,
            "invalid components for `cell`: there should be 2 components, got "
            + std::to_string(components.size())
        );
    }

    if (*components[0] != metatensor::Labels({"cell_abc"}, {{0}, {1}, {2}})) {
        C10_THROW_ERROR(ValueError,
            "invalid components for `cell`: the first component should be 'cell_abc'=[0, 1, 2]"
        );
    }

    if (*components[1] != metatensor::Labels({"xyz"}, {{0}, {1}, {2}})) {
        C10_THROW_ERROR(ValueError,
            "invalid components for `cell`: the second component should be 'xyz'=[0, 1, 2]"
        );
    }

    if (*cell->properties() != metatensor::Labels({"cell"}, {{0}})) {
        C10_THROW_ERROR(ValueError,
            "invalid properties for `cell`: there should be a single 'cell'=0 property"
        );
    }

    if (!cell->gradients_list().empty()) {
        C10_THROW_ERROR(ValueError, "`cell` should not have any gradients");
    }
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

    // actually add the neighbors lists
    auto it = neighbors_.find(options);
    if (it != neighbors_.end()) {
        C10_THROW_ERROR(ValueError,
            "the neighbors list for " + options->repr() + " already exists in this system"
        );
    }

    neighbors_.emplace(std::move(options), std::move(neighbors));
}

TorchTensorBlock SystemHolder::get_neighbors_list(NeighborsListOptions options) const {
    auto it = neighbors_.find(options);
    if (it == neighbors_.end()) {
        C10_THROW_ERROR(ValueError,
            "No neighbors list for " + options->repr() + " was found.\n"
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


void SystemHolder::add_data(std::string name, TorchTensorBlock values) {
    if (name == "positions" || name == "cell" || name == "neighbors") {
        C10_THROW_ERROR(ValueError, "custom data can not be 'positions', 'cell', or 'neighbors'");
    }

    if (data_.find(name) != data_.end()) {
        C10_THROW_ERROR(ValueError, "custom data for '" + name + "' is already present in this system");
    }

    data_.emplace(std::move(name), std::move(values));
}

TorchTensorBlock SystemHolder::get_data(std::string name) const {
    if (name == "positions") {
        return positions;
    } else if (name == "cell") {
        return cell;
    } else {
        auto it = data_.find(name);
        if (it == data_.end()) {
            C10_THROW_ERROR(ValueError,
                "no data for '" + name + "' found in this system"
            );
        }

        TORCH_WARN_ONCE("custom data (", name,") is experimental, please contact the developers to add your data in the main API");
        return it->second;
    }
}


std::vector<std::string> SystemHolder::known_data() const {
    auto result = std::vector<std::string>();
    for (const auto& it: data_) {
        result.emplace_back(it.first);
    }
    return result;
}
