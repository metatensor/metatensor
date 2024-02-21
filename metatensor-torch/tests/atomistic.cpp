#include <torch/torch.h>

#include <metatensor/torch.hpp>
#include <metatensor/torch/atomistic.hpp>
using namespace metatensor_torch;

#include <catch.hpp>
using namespace Catch::Matchers;

TEST_CASE("Models metadata") {
    SECTION("NeighborsListOptions") {
        // save to JSON
        auto options = torch::make_intrusive<NeighborsListOptionsHolder>(3.5426, true);
        const auto* expected = R"({
    "class": "NeighborsListOptions",
    "full_list": true,
    "model_cutoff": 4615159644819978768
})";
        CHECK(options->to_json() == expected);

        // load from JSON
        std::string json = R"({
    "model_cutoff": 4615159644819978768,
    "full_list": false,
    "class": "NeighborsListOptions"
})";
        options = NeighborsListOptionsHolder::from_json(json);
        CHECK(options->model_cutoff() == 3.5426);
        CHECK(options->full_list() == false);

        CHECK_THROWS_WITH(
            NeighborsListOptionsHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for NeighborsListOptions, did not find it")
        );
        CHECK_THROWS_WITH(
            NeighborsListOptionsHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for NeighborsListOptions must be 'NeighborsListOptions'")
        );
    }

    SECTION("ModelOutput") {
        // save to JSON
        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->quantity = "foo";
        output->unit = "bar";
        output->per_atom = false;
        output->explicit_gradients = {"baz", "not.this-one_"};

        const auto* expected = R"({
    "class": "ModelOutput",
    "explicit_gradients": [
        "baz",
        "not.this-one_"
    ],
    "per_atom": false,
    "quantity": "foo",
    "unit": "bar"
})";
        CHECK(output->to_json() == expected);

        // load from JSON
        std::string json = R"({
    "class": "ModelOutput",
    "quantity": "quantity",
    "explicit_gradients": []
})";
        output = ModelOutputHolder::from_json(json);
        CHECK(output->quantity == "quantity");
        CHECK(output->unit.empty());
        CHECK(output->per_atom == false);
        CHECK(output->explicit_gradients.empty());

        CHECK_THROWS_WITH(
            ModelOutputHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelOutput, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelOutputHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelOutput must be 'ModelOutput'")
        );
    }

    SECTION("ModelEvaluationOptions") {
        // save to JSON
        auto options = torch::make_intrusive<ModelEvaluationOptionsHolder>();
        options->length_unit = "mm";

        options->outputs.insert("output_1", torch::make_intrusive<ModelOutputHolder>());

        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->per_atom = true;
        output->unit = "something";
        options->outputs.insert("output_2", output);

        const auto* expected = R"({
    "class": "ModelEvaluationOptions",
    "length_unit": "mm",
    "outputs": {
        "output_1": {
            "class": "ModelOutput",
            "explicit_gradients": [],
            "per_atom": false,
            "quantity": "",
            "unit": ""
        },
        "output_2": {
            "class": "ModelOutput",
            "explicit_gradients": [],
            "per_atom": true,
            "quantity": "",
            "unit": "something"
        }
    },
    "selected_atoms": null
})";
        CHECK(options->to_json() == expected);


        // load from JSON
        std::string json =R"({
    "length_unit": "very large",
    "outputs": {
        "foo": {
            "explicit_gradients": ["test"],
            "class": "ModelOutput"
        }
    },
    "selected_atoms": {
        "names": ["system", "atom"],
        "values": [0, 1, 4, 5]
    },
    "class": "ModelEvaluationOptions"
})";

        options = ModelEvaluationOptionsHolder::from_json(json);
        CHECK(options->length_unit == "very large");
        auto expected_selection = LabelsHolder::create(
            {"system", "atom"},
            {{0, 1}, {4, 5}}
        );
        CHECK(*options->get_selected_atoms().value() == *expected_selection);

        output = options->outputs.at("foo");
        CHECK(output->quantity.empty());
        CHECK(output->unit.empty());
        CHECK(output->per_atom == false);
        CHECK(output->explicit_gradients == std::vector<std::string>{"test"});

        CHECK_THROWS_WITH(
            ModelEvaluationOptionsHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelEvaluationOptions, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelEvaluationOptionsHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelEvaluationOptions must be 'ModelEvaluationOptions'")
        );
    }

    SECTION("ModelCapabilities") {
        // save to JSON
        auto capabilities = torch::make_intrusive<ModelCapabilitiesHolder>();
        capabilities->length_unit = "µm";
        capabilities->interaction_range = 1.4;
        capabilities->atomic_types = {1, 2, -43};
        capabilities->supported_devices = {"cuda", "xla", "cpu"};

        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->per_atom = true;
        output->quantity = "something";
        capabilities->outputs.insert("bar", output);

        const auto* expected = R"({
    "atomic_types": [
        1,
        2,
        -43
    ],
    "class": "ModelCapabilities",
    "interaction_range": 4608983858650965606,
    "length_unit": "\u00b5m",
    "outputs": {
        "bar": {
            "class": "ModelOutput",
            "explicit_gradients": [],
            "per_atom": true,
            "quantity": "something",
            "unit": ""
        }
    },
    "supported_devices": [
        "cuda",
        "xla",
        "cpu"
    ]
})";
        CHECK(capabilities->to_json() == expected);


        // load from JSON
        std::string json =R"({
    "length_unit": "\u00b5m",
    "outputs": {
        "foo": {
            "explicit_gradients": ["test"],
            "class": "ModelOutput"
        }
    },
    "atomic_types": [
        1,
        -2
    ],
    "class": "ModelCapabilities"
})";

        capabilities = ModelCapabilitiesHolder::from_json(json);
        CHECK(capabilities->length_unit == "µm");
        CHECK(capabilities->atomic_types == std::vector<int64_t>{1, -2});

        output = capabilities->outputs.at("foo");
        CHECK(output->quantity.empty());
        CHECK(output->unit.empty());
        CHECK(output->per_atom == false);
        CHECK(output->explicit_gradients == std::vector<std::string>{"test"});

        CHECK_THROWS_WITH(
            ModelCapabilitiesHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelCapabilities, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelCapabilitiesHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelCapabilities must be 'ModelCapabilities'")
        );
    }
}
