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

class ModelMetadataHolder;
/// TorchScript will always manipulate `ModelMetadataHolder` through a `torch::intrusive_ptr`
using ModelMetadata = torch::intrusive_ptr<ModelMetadataHolder>;



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
    static ModelOutput from_json(std::string_view json);
};


/// Description of a model's capabilities, i.e. everything a model can do.
class METATENSOR_TORCH_EXPORT ModelCapabilitiesHolder: public torch::CustomClassHolder {
public:
    ModelCapabilitiesHolder() = default;

    /// Initialize `ModelCapabilities` with the given data
    ModelCapabilitiesHolder(
        torch::Dict<std::string, ModelOutput> outputs_,
        std::vector<int64_t> atomic_types_,
        double interaction_range_,
        std::string length_unit_,
        std::vector<std::string> supported_devices_
    ):
        outputs(outputs_),
        atomic_types(std::move(atomic_types_)),
        interaction_range(interaction_range_),
        length_unit(std::move(length_unit_)),
        supported_devices(std::move(supported_devices_))
    {}

    ~ModelCapabilitiesHolder() override = default;

    /// all possible outputs from this model and corresponding settings
    torch::Dict<std::string, ModelOutput> outputs;

    /// which types the model can handle
    std::vector<int64_t> atomic_types;

    /// How far a given atom needs to know about other atoms, in the length unit
    /// of the model.
    ///
    /// This is used to properly implement domain decomposition with this model.
    ///
    /// For a short range model, this is the same as the largest neighbor list
    /// cutoff. For a message passing model, this is the cutoff of one
    /// environment times the number of message passing steps. For an explicit
    /// long range model, this should be set to infinity.
    ///
    /// This will default to -1 if not explicitly set by the user.
    double interaction_range = -1;

    /// unit of lengths the model expects as input
    std::string length_unit;

    /// Same as `interaction_range`, but in the length unit of the engine
    double engine_interaction_range() const {
        return interaction_range * model_to_engine_;
    }

    /// Set the conversion factor from the model length units to the engine
    /// units.
    ///
    /// This should be called before `engine_interaction_range()`.
    void set_engine_unit(double conversion) {
        model_to_engine_ = conversion;
    }

    /// What devices can this model run on? This should only contain the
    /// `device_type` part of the device, and not the device number (i.e. this
    /// should be `"cuda"`, not `"cuda:0"`).
    ///
    /// Devices should be ordered in order of preference: first one should be
    /// the best device for this model, and so on.
    std::vector<std::string> supported_devices;

    /// Serialize a `ModelCapabilities` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelCapabilities` from a JSON string.
    static ModelCapabilities from_json(std::string_view json);

private:
    double model_to_engine_ = 1.0;
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
    static ModelEvaluationOptions from_json(std::string_view json);

private:
    torch::optional<TorchLabels> selected_atoms_ = torch::nullopt;
};


/// Metadata about a specific exported model
class METATENSOR_TORCH_EXPORT ModelMetadataHolder: public torch::CustomClassHolder {
public:
    ModelMetadataHolder() = default;

    /// Initialize `ModelMetadata` with the given information
    ModelMetadataHolder(
        std::string name_,
        std::string description_,
        std::vector<std::string> authors_,
        torch::Dict<std::string, std::vector<std::string>> references_
    ):
        name(std::move(name_)),
        description(std::move(description_)),
        authors(std::move(authors_)),
        references(references_)
    {
        this->validate();
    }

    ~ModelMetadataHolder() override = default;

    /// Name of this model
    std::string name;

    /// Description of this model
    std::string description;

    /// List of authors for this model
    std::vector<std::string> authors;

    /// References for this model. The top level dict can have three keys:
    ///
    /// - "implementation": for reference to software and libraries used in the
    ///   implementation of the model (i.e. for PyTorch
    ///   https://dl.acm.org/doi/10.5555/3454287.3455008)
    /// - "architecture": for reference that introduced the general architecture
    ///   used by this model
    /// - "model": for reference specific to this exact model
    torch::Dict<std::string, std::vector<std::string>> references;

    /// Implementation of Python's `__repr__` and `__str__`, printing all
    /// metadata about this model.
    std::string print() const;

    /// Serialize `ModelMetadata` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelMetadata` from a JSON string.
    static ModelMetadata from_json(std::string_view json);

private:
    /// validate the metadata before using it
    void validate() const;
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
