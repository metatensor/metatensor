#ifndef METATENSOR_TORCH_ATOMISTIC_MODEL_HPP
#define METATENSOR_TORCH_ATOMISTIC_MODEL_HPP

#include <vector>
#include <string>

#include <torch/script.h>

#include <metatensor.hpp>

#include "metatensor/torch/labels.hpp"
#include "metatensor/torch/exports.h"

namespace metatensor_torch {

class ModelOutputHolder;
/// TorchScript will always manipulate `ModelOutputHolder` through a `torch::intrusive_ptr`
using ModelOutput = torch::intrusive_ptr<ModelOutputHolder>;

class ModelCapabilitiesHolder;
/// TorchScript will always manipulate `ModelCapabilitiesHolder` through a `torch::intrusive_ptr`
using ModelCapabilities = torch::intrusive_ptr<ModelCapabilitiesHolder>;

class ModelEvaluationOptionsHolder;
/// TorchScript will always manipulate `ModelEvaluationOptionsHolder` through a `torch::intrusive_ptr`
using ModelEvaluationOptions = torch::intrusive_ptr<ModelEvaluationOptionsHolder>;


/// Description of one of the quantity a model can compute
class METATENSOR_TORCH_EXPORT ModelOutputHolder: public torch::CustomClassHolder {
public:
    ModelOutputHolder() = default;

    /// Initialize `ModelOutput` with the given data
    ModelOutputHolder(
        std::string quantity_,
        std::string unit_,
        bool per_atom_,
        std::vector<std::string> explicit_gradients_
    ):
        quantity(std::move(quantity_)),
        unit(std::move(unit_)),
        per_atom(per_atom_),
        explicit_gradients(std::move(explicit_gradients_))
    {}

    ~ModelOutputHolder() override = default;

    /// quantity of the output (e.g. energy, dipole, …).  If this is an empty
    /// string, no unit conversion will be performed.
    std::string quantity;

    /// unit of the output. If this is an empty string, no unit conversion will
    /// be performed.
    std::string unit;

    /// is the output defined per-atom or for the overall structure
    bool per_atom = false;

    /// Which gradients should be computed eagerly and stored inside the output
    /// `TensorMap`
    std::vector<std::string> explicit_gradients;

    /// Serialize a `ModelOutput` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelOutput` from a JSON string.
    static ModelOutput from_json(const std::string& json);
};


/// Description of a model's capabilities, i.e. everything a model can do.
class METATENSOR_TORCH_EXPORT ModelCapabilitiesHolder: public torch::CustomClassHolder {
public:
    ModelCapabilitiesHolder() = default;

    /// Initialize `ModelCapabilities` with the given data
    ModelCapabilitiesHolder(
        std::string length_unit_,
        std::vector<int64_t> types_,
        torch::Dict<std::string, ModelOutput> outputs_
    ):
        length_unit(std::move(length_unit_)),
        types(std::move(types_)),
        outputs(outputs_)
    {}

    ~ModelCapabilitiesHolder() override = default;

    /// unit of lengths the model expects as input
    std::string length_unit;

    /// which particle types the model can handle
    std::vector<int64_t> types;

    /// all possible outputs from this model and corresponding settings
    torch::Dict<std::string, ModelOutput> outputs;

    /// Serialize a `ModelCapabilities` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelCapabilities` from a JSON string.
    static ModelCapabilities from_json(const std::string& json);
};


/// Options requested by the simulation engine when running with a model
class METATENSOR_TORCH_EXPORT ModelEvaluationOptionsHolder: public torch::CustomClassHolder {
public:
    ModelEvaluationOptionsHolder() = default;

    /// Initialize `ModelEvaluationOptions` with the given data
    ModelEvaluationOptionsHolder(
        std::string length_unit,
        torch::Dict<std::string, ModelOutput> outputs,
        torch::optional<TorchLabels> selected_atoms
    );

    ~ModelEvaluationOptionsHolder() override = default;

    /// unit of lengths the engine uses for the model input
    std::string length_unit;

    /// requested outputs for this run and corresponding settings
    torch::Dict<std::string, ModelOutput> outputs;

    /// Only run the calculation for a selected subset of atoms. If this is set
    /// to `None`, run the calculation on all atoms. If this is a set of
    /// `Labels`, it will have two dimensions named `"system"` and `"atom"`,
    /// containing the 0-based indices of all the atoms in the selected subset.
    torch::optional<TorchLabels> get_selected_atoms() const {
        return selected_atoms_;
    }

    /// Setter for `selected_atoms`
    void set_selected_atoms(torch::optional<TorchLabels> selected_atoms);

    /// Serialize a `ModelEvaluationOptions` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelEvaluationOptions` from a JSON string.
    static ModelEvaluationOptions from_json(const std::string& json);

private:
    torch::optional<TorchLabels> selected_atoms_ = torch::nullopt;
};

/// Check the exported metatensor atomistic model at the given `path`, and
/// warn/error as required.
METATENSOR_TORCH_EXPORT void check_atomistic_model(std::string path);

/// Check and then load the metatensor atomistic model at the given `path`.
METATENSOR_TORCH_EXPORT torch::jit::Module load_atomistic_model(
    std::string path,
    c10::optional<c10::Device> device = c10::nullopt
);

}

#endif
