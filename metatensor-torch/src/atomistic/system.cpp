#include <cctype>
#include <cstring>

#include <sstream>
#include <algorithm>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <metatensor.hpp>

#include "metatensor/torch/atomistic/system.hpp"
#include "metatensor/torch/atomistic/model.hpp"

#include "../internal/utils.hpp"


using namespace metatensor_torch;

NeighborListOptionsHolder::NeighborListOptionsHolder(
    double cutoff,
    bool full_list,
    bool strict,
    std::string requestor
):
    cutoff_(cutoff),
    full_list_(full_list),
    strict_(strict)
{
    this->add_requestor(std::move(requestor));
}


void NeighborListOptionsHolder::add_requestor(std::string requestor) {
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

void NeighborListOptionsHolder::set_length_unit(std::string length_unit) {
    validate_unit("length", length_unit);
    this->length_unit_ = std::move(length_unit);
}

double NeighborListOptionsHolder::engine_cutoff(const std::string& engine_length_unit) const {
    return cutoff_ * unit_conversion_factor("length", length_unit_, engine_length_unit);
}

std::string NeighborListOptionsHolder::repr() const {
    auto ss = std::ostringstream();

    ss << "NeighborListOptions\n";
    ss << "    cutoff: " << std::to_string(cutoff_);
    if (!length_unit_.empty()) {
        ss << " " << length_unit_;
    }
    ss << "\n    full_list: " << (full_list_ ? "True" : "False") << "\n";
    ss << "    strict: " << (strict_ ? "True" : "False") << "\n";

    if (!requestors_.empty()) {
        ss << "    requested by:\n";
        for (const auto& requestor: requestors_) {
            ss << "        - " << requestor << "\n";
        }
    }

    return ss.str();
}

std::string NeighborListOptionsHolder::str() const {
    return "NeighborListOptions(cutoff=" + std::to_string(cutoff_) + \
        ", full_list=" + (full_list_ ? "True" : "False") + \
        ", strict=" + (strict_ ? "True" : "False") + ")";
}

std::string NeighborListOptionsHolder::to_json() const {
    nlohmann::json result;

    result["class"] = "NeighborListOptions";

    // Store cutoff using it's binary representation to ensure perfect
    // round-tripping of the data
    static_assert(sizeof(double) == sizeof(int64_t));
    int64_t int_cutoff = 0;
    std::memcpy(&int_cutoff, &this->cutoff_, sizeof(double));
    result["cutoff"] = int_cutoff;
    result["full_list"] = this->full_list_;
    result["strict"] = this->strict_;
    result["length_unit"] = this->length_unit_;

    result["requestors"] = this->requestors_;

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

NeighborListOptions NeighborListOptionsHolder::from_json(const std::string& json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for NeighborListOptions, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for NeighborListOptions, did not find it");
    }

    if (data["class"] != "NeighborListOptions") {
        throw std::runtime_error("'class' in JSON for NeighborListOptions must be 'NeighborListOptions'");
    }

    if (!data.contains("cutoff") || !data["cutoff"].is_number_integer()) {
        throw std::runtime_error("'cutoff' in JSON for NeighborListOptions must be a number");
    }
    auto int_cutoff = data["cutoff"].get<int64_t>();
    double cutoff = 0;
    std::memcpy(&cutoff, &int_cutoff, sizeof(double));

    if (!data.contains("full_list") || !data["full_list"].is_boolean()) {
        throw std::runtime_error("'full_list' in JSON for NeighborListOptions must be a boolean");
    }
    auto full_list = data["full_list"].get<bool>();

    if (!data.contains("strict") || !data["strict"].is_boolean()) {
        throw std::runtime_error("'strict' in JSON for NeighborListOptions must be a boolean");
    }
    auto strict = data["strict"].get<bool>();

    auto options = torch::make_intrusive<NeighborListOptionsHolder>(cutoff, full_list, strict);

    if (data.contains("length_unit")) {
        if (!data["length_unit"].is_string()) {
            throw std::runtime_error("'length_unit' in JSON for NeighborListOptions must be a string");
        }
        options->set_length_unit(data["length_unit"]);
    }

    if (data.contains("requestors")) {
        if (!data["requestors"].is_array()) {
            throw std::runtime_error("'requestors' in JSON for NeighborListOptions must be an array");
        }

        for (const auto& requestor: data["requestors"]) {
            if (!requestor.is_string()) {
                throw std::runtime_error("'requestors' in JSON for NeighborListOptions must be an array of strings");
            }

            options->add_requestor(requestor.get<std::string>());
        }
    }

    return options;
}

// ========================================================================== //

torch::Tensor NeighborsAutograd::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor positions,
    torch::Tensor cell,
    TensorBlock neighbors,
    bool check_consistency
) {
    auto distances = neighbors->values();

    if (check_consistency) {
        auto n_atoms = positions.size(0);
        auto epsilon = 1e-6;
        if (distances.scalar_type() != torch::kFloat64) {
            epsilon = 1e-4;
        }

        auto samples = neighbors->samples()->values();
        for (int64_t sample_i=0; sample_i<samples.size(0); sample_i++) {
            auto atom_i = samples[sample_i][0];
            auto atom_j = samples[sample_i][1];

            auto atom_i_cpu = atom_i.to(torch::kCPU).item<int32_t>();
            auto atom_j_cpu = atom_j.to(torch::kCPU).item<int32_t>();
            if (atom_i_cpu < 0 || atom_i_cpu >= n_atoms) {
                C10_THROW_ERROR(ValueError,
                    "checking internal consistency: 'first_atom' in neighbor list (" +
                    std::to_string(atom_i_cpu) + ") is out of bounds (we have " +
                    std::to_string(n_atoms) + " atoms in the system)"
                );
            }

            if (atom_j_cpu < 0 || atom_j_cpu >= n_atoms) {
                C10_THROW_ERROR(ValueError,
                    "checking internal consistency: 'second_atom' in neighbor list (" +
                    std::to_string(atom_j_cpu) + ") is out of bounds (we have " +
                    std::to_string(n_atoms) + " atoms in the system)"
                );
            }

            auto cell_shift = samples.index({sample_i, torch::indexing::Slice(2, 5)}).to(positions.scalar_type());

            auto actual_distance = distances[sample_i].reshape({3});
            auto expected_distance = positions[atom_j] - positions[atom_i] + cell_shift.matmul(cell);

            auto diff_norm = (actual_distance - expected_distance).norm().to(torch::kCPU).to(torch::kF64);
            if (diff_norm.item<double>() > epsilon) {
                std::ostringstream oss;

                oss << "checking internal consistency: one neighbor pair does not match its metadata: ";
                oss << "the pair between atom " << atom_i.item<int32_t>();
                oss << " and atom " << atom_j.item<int32_t>() << " for the ";

                auto cell_shift_i32 = samples.index({sample_i, torch::indexing::Slice(2, 5)});
                oss << "[" << cell_shift_i32[0].item<int32_t>() << ", ";
                oss << cell_shift_i32[1].item<int32_t>() << ", ";
                oss << cell_shift_i32[2].item<int32_t>() << "] cell shift ";

                auto expected_f64 = expected_distance.to(torch::kCPU).to(torch::kF64);
                oss << "should have a distance vector of ";
                oss << "[" << expected_f64[0].item<double>() << ", ";
                oss << expected_f64[1].item<double>() << ", ";
                oss << expected_f64[2].item<double>() << "] ";


                auto actual_f64 = actual_distance.to(torch::kCPU).to(torch::kF64);
                oss << "but has a distance vector of ";
                oss << "[" << actual_f64[0].item<double>() << ", ";
                oss << actual_f64[1].item<double>() << ", ";
                oss << actual_f64[2].item<double>() << "] ";

                oss << "norm difference is " << diff_norm.item<double>();

                C10_THROW_ERROR(ValueError, oss.str());
            }
        }
    }

    ctx->save_for_backward({positions, cell, neighbors->values(), neighbors->samples()->values()});

    return distances;
}


std::vector<torch::Tensor> NeighborsAutograd::backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<torch::Tensor> outputs_grad
) {
    const auto& distances_grad = outputs_grad[0];

    const auto& saved_variables = ctx->get_saved_variables();
    const auto& positions = saved_variables[0];
    const auto& cell = saved_variables[1];

    const auto& distances = saved_variables[2];
    const auto& samples = saved_variables[3];

    auto positions_grad = torch::Tensor();
    if (positions.requires_grad()) {
        positions_grad = torch::zeros_like(positions);
        positions_grad = torch::index_add(
            positions_grad,
            /*dim=*/0,
            /*index=*/samples.index({torch::indexing::Slice(), 1}),
            /*source=*/distances_grad.squeeze(-1),
            /*alpha=*/1.0
        );
        positions_grad = torch::index_add(
            positions_grad,
            /*dim=*/0,
            /*index=*/samples.index({torch::indexing::Slice(), 0}),
            /*source=*/distances_grad.squeeze(-1),
            /*alpha=*/-1.0
        );
    }

    auto cell_grad = torch::Tensor();
    if (cell.requires_grad()) {
        auto cell_shifts = samples.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(2, 5)
        }).to(cell.scalar_type());
        cell_grad = cell_shifts.t().matmul(distances_grad.squeeze(-1));
    }

    return {positions_grad, cell_grad, torch::Tensor(), torch::Tensor()};
}

void metatensor_torch::register_autograd_neighbors(
    System system,
    TensorBlock neighbors,
    bool check_consistency
) {
    auto distances = neighbors->values();
    if (distances.requires_grad()) {
        C10_THROW_ERROR(ValueError,
            "`neighbors` is already part of a computational graph, "
            "detach it before calling `register_autograd_neighbors()`"
        );
    }

    // these checks should be fine in a normal use case, but might be false if
    // someone gives weird data to the function. `check_consistency=True` should
    // help debug this kind of issues.
    if (check_consistency) {
        if (system->positions().device() != distances.device()) {
            C10_THROW_ERROR(ValueError,
                "`system` and `neighbors` must be on the same device, "
                "got " + system->positions().device().str() + " and " +
                distances.device().str()
            );
        }

        if (system->positions().scalar_type() != distances.scalar_type()) {
            C10_THROW_ERROR(ValueError,
                "`system` and `neighbors` must have the same dtype, "
                "got " + scalar_type_name(system->positions().scalar_type()) +
                " and " + scalar_type_name(distances.scalar_type())
            );
        }

        auto expected_names = std::vector<std::string>{
            "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
        };
        if (neighbors->samples()->names() != expected_names) {
            C10_THROW_ERROR(ValueError,
                "invalid `neighbors`: expected sample names to be ['first_atom', "
                "'second_atom', 'cell_shift_a', 'cell_shift_b', 'cell_shift_c']"
            );
        }

        expected_names = std::vector<std::string>{"xyz"};
        if (neighbors->components().size() != 1 || neighbors->components()[0]->names() != expected_names) {
            C10_THROW_ERROR(ValueError,
                "invalid `neighbors`: expected component names to be ['xyz']"
            );
        }

        expected_names = std::vector<std::string>{"distance"};
        if (neighbors->properties()->names() != expected_names) {
            C10_THROW_ERROR(ValueError,
                "invalid `neighbors`: expected property names to be ['distance']"
            );
        }
    }

    // WARNING: we are not using `torch::autograd::Function` in the usual way.
    // Instead we pass already computed data (in `neighbors->values()`), and
    // directly return the corresponding tensor in `forward()`. This still
    // causes torch to register the right function for `backward`.
    auto _ = NeighborsAutograd::apply(
        system->positions(),
        system->cell(),
        neighbors,
        check_consistency
    );
}

// ========================================================================== //

static bool is_floating_point(torch::Dtype dtype) {
    return dtype == torch::kF16 || dtype == torch::kF32 || dtype == torch::kF64;
}

SystemHolder::SystemHolder(torch::Tensor types, torch::Tensor positions, torch::Tensor cell, torch::Tensor pbc):
    types_(std::move(types)),
    positions_(std::move(positions)),
    cell_(std::move(cell)),
    pbc_(std::move(pbc))
{
    if (positions_.device() != types_.device() || cell_.device() != types_.device() || pbc_.device() != types_.device()) {
        C10_THROW_ERROR(ValueError,
            "`types`, `positions`, `cell`, and `pbc` must be on the same device, got " +
            types_.device().str() + ", " + positions_.device().str() + ", " +
            cell_.device().str() + ", and " + pbc_.device().str()
        );
    }

    if (types_.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError,
            "`types` must be a 1 dimensional tensor, got a tensor with " +
            std::to_string(types_.sizes().size()) + " dimensions"
        );
    }

    if (torch::canCast(types_.scalar_type(), torch::kInt32)) {
        types_ = types_.to(torch::kInt32);
    } else {
        C10_THROW_ERROR(ValueError,
            "`types` must be a tensor of integers, got " +
            scalar_type_name(types_.scalar_type()) + " instead"
        );
    }

    auto n_atoms = types_.size(0);
    if (positions_.sizes().size() != 2) {
        C10_THROW_ERROR(ValueError,
            "`positions` must be a 2 dimensional tensor, got a tensor with " +
            std::to_string(positions_.sizes().size()) + " dimensions"
        );
    }

    if (positions_.size(0) != n_atoms || positions_.size(1) != 3) {
        C10_THROW_ERROR(ValueError,
            "`positions` must be a (len(types) x 3) tensor, got a tensor with shape [" +
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

    if (pbc_.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError,
            "`pbc` must be a 1 dimensional tensor, got a tensor with " +
            std::to_string(pbc_.sizes().size()) + " dimensions"
        );
    }

    if (pbc_.size(0) != 3) {
        C10_THROW_ERROR(ValueError,
            "`pbc` must contain 3 entries, got a tensor with " +
            std::to_string(pbc_.size(0)) + " values"
        );
    }

    if (pbc_.scalar_type() != torch::kBool) {
        C10_THROW_ERROR(ValueError,
            "`pbc` must be a tensor of booleans, got " +
            scalar_type_name(pbc_.scalar_type()) + " instead"
        );
    }

    // if the PBC are False along any directions, we check that the
    // corresponding cell vectors are zero
    if (!pbc_.device().is_meta()) {
        auto cell_where_pbc_is_false = cell_.index({pbc_ == false});
        if (!torch::all(cell_where_pbc_is_false == 0.0).item<bool>()) {
            C10_THROW_ERROR(ValueError,
                "if `pbc` is False along any direction, the corresponding cell vector must be zero"
            );
        }
    }
}


void SystemHolder::set_types(torch::Tensor types) {
    if (types.device() != this->device()) {
        C10_THROW_ERROR(ValueError,
            "new `types` must be on the same device as existing data, got " +
            types.device().str() + " and " + this->device().str()
        );
    }

    if (types.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError,
            "new `types` must be a 1 dimensional tensor, got a tensor with " +
            std::to_string(types.sizes().size()) + " dimensions"
        );
    }

    if (types.size(0) != this->size()) {
        C10_THROW_ERROR(ValueError,
            "new `types` must contain " + std::to_string(this->size()) + " entries, "
            "got a tensor with " + std::to_string(types.size(0)) + " values"
        );
    }

    if (torch::canCast(types.scalar_type(), torch::kInt32)) {
        types = types.to(torch::kInt32);
    } else {
        C10_THROW_ERROR(ValueError,
            "new `types` must be a tensor of integers, got " +
            scalar_type_name(types.scalar_type()) + " instead"
        );
    }

    this->types_ = std::move(types);
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
            "new `positions` must be a (len(types) x 3) tensor, got a tensor with shape [" +
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

void SystemHolder::set_pbc(torch::Tensor pbc) {
    if (pbc.device() != this->device()) {
        C10_THROW_ERROR(ValueError,
            "new `pbc` must be on the same device as existing data, got " +
            pbc.device().str() + " and " + this->device().str()
        );
    }

    if (pbc.scalar_type() != torch::kBool) {
        C10_THROW_ERROR(ValueError,
            "new `pbc` must be a tensor of booleans, got " +
            scalar_type_name(pbc.scalar_type()) + " instead"
        );
    }

    if (pbc.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError,
            "new `pbc` must be a 1 dimensional tensor, got a tensor with " +
            std::to_string(pbc.sizes().size()) + " dimensions"
        );
    }

    if (pbc.size(0) != 3) {
        C10_THROW_ERROR(ValueError,
            "new `pbc` must contain 3 entries, got a tensor with " +
            std::to_string(pbc.size(0)) + " values"
        );
    }


    // if the PBC are False along any directions, we check that the
    // corresponding cell vectors are zero
    if (!pbc_.device().is_meta()) {
        auto cell_where_pbc_is_false = cell_.index({pbc == false});
        if (!torch::all(cell_where_pbc_is_false == 0.0).item<bool>()) {
            C10_THROW_ERROR(ValueError,
                "if `pbc` is False along any direction, the corresponding cell vector must be zero"
            );
        }
    }

    this->pbc_ = std::move(pbc);
}

System SystemHolder::to(
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device
) const {
    auto system = torch::make_intrusive<SystemHolder>(
        this->types().to(
            /*dtype*/ torch::nullopt,
            /*layout*/ torch::nullopt,
            device,
            /*pin_memory*/ torch::nullopt,
            /*non_blocking*/ false,
            /*copy*/ false,
            /*memory_format*/ torch::MemoryFormat::Preserve
        ),
        this->positions().to(
            dtype,
            /*layout*/ torch::nullopt,
            device,
            /*pin_memory*/ torch::nullopt,
            /*non_blocking*/ false,
            /*copy*/ false,
            /*memory_format*/ torch::MemoryFormat::Preserve
        ),
        this->cell().to(
            dtype,
            /*layout*/ torch::nullopt,
            device,
            /*pin_memory*/ torch::nullopt,
            /*non_blocking*/ false,
            /*copy*/ false,
            /*memory_format*/ torch::MemoryFormat::Preserve
        ),
        this->pbc().to(
            /*dtype*/ torch::nullopt,
            /*layout*/ torch::nullopt,
            device,
            /*pin_memory*/ torch::nullopt,
            /*non_blocking*/ false,
            /*copy*/ false,
            /*memory_format*/ torch::MemoryFormat::Preserve
        )
    );

    for (const auto& it: this->neighbors_) {
        system->add_neighbor_list(it.first, it.second->to(dtype, device));
    }

    for (const auto& it: this->data_) {
        system->add_data(it.first, it.second->to(dtype, device));
    }

    return system;
}


System SystemHolder::to_positional(
    torch::IValue positional_1,
    torch::IValue positional_2,
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device
) const {
    auto [parsed_dtype, parsed_device] = to_arguments_parse(
        positional_1,
        positional_2,
        dtype,
        device,
        "`System.to`"
    );

    return this->to(parsed_dtype, parsed_device);
}


void SystemHolder::add_neighbor_list(NeighborListOptions options, TensorBlock neighbors) {
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

    auto requires_grad = positions_.requires_grad() || cell_.requires_grad();
    if (requires_grad && !neighbors->values().requires_grad()) {
        TORCH_WARN(
            "This system's positions or cell requires grad, but the neighbors does not. ",
            "You should use `register_autograd_neighbors()` to make sure the neighbors "
            "distance vectors are integrated in the computational graph."
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

TensorBlock SystemHolder::get_neighbor_list(NeighborListOptions options) const {
    auto it = neighbors_.find(options);
    if (it == neighbors_.end()) {
        C10_THROW_ERROR(ValueError,
            "No neighbor list for " + options->str() + " was found.\n"
            "Is it part of the `requested_neighbor_lists` for this model?"
        );
    }
    return it->second;
}

std::vector<NeighborListOptions> SystemHolder::known_neighbor_lists() const {
    auto result = std::vector<NeighborListOptions>();
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
    "types",
    "positions", "position",
    "cell",
    "neighbors", "neighbor"
};

void SystemHolder::add_data(std::string name, TensorMap tensor, bool override) {
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

    if (!override && data_.find(name) != data_.end()) {
        C10_THROW_ERROR(ValueError,
            "custom data '" + name + "' is already present in this system"
        );
    }

    if (tensor->device() != this->device()) {
        C10_THROW_ERROR(ValueError,
            "device (" + tensor->device().str() + ") of the custom data "
            "'" + name + "' does not match this system device (" + this->device().str() +")"
        );
    }

    if (tensor->scalar_type() != this->scalar_type()) {
        C10_THROW_ERROR(ValueError,
            "dtype (" + scalar_type_name(tensor->scalar_type()) + ") of " +
            "custom data '" + name + "' does not match this system " +
            "dtype (" + scalar_type_name(this->scalar_type()) +")"
        );
    }

    data_.insert_or_assign(std::move(name), std::move(tensor));
}

TensorMap SystemHolder::get_data(std::string name) const {
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

    static std::unordered_set<std::string> ALREADY_WARNED = {};
    if (ALREADY_WARNED.insert(name).second) {
        TORCH_WARN(
            "custom data '", name, "' is experimental, please contact metatensor's ",
            "developers to add this data as a member of the `System` class"
        );
    }

    return it->second;
}


std::vector<std::string> SystemHolder::known_data() const {
    auto result = std::vector<std::string>();
    for (const auto& it: data_) {
        result.emplace_back(it.first);
    }
    return result;
}

std::string SystemHolder::str() const {
    auto result = std::ostringstream();
    result << "System with " << this->size() << " atoms, ";

    auto cell = cell_.to(torch::kCPU, torch::kF64);
    if (torch::all(cell == torch::zeros_like(cell)).item<bool>()) {
        result << "non periodic";
    } else {
        result << "periodic cell: [";
        for (int64_t i=0; i<3; i++) {
            for (int64_t j=0; j<3; j++) {
                result << cell_.index({i, j}).item<double>();
                if (j != 2 || i != 2) {
                    result << ", ";
                }
            }
        }
        result << "]";
    }

    return result.str();
}
