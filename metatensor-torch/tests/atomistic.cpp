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
    "full_list": true,
    "model_cutoff": 4615159644819978768,
    "type": "NeighborsListOptions"
})";
        CHECK(options->to_json() == expected);

        // load from JSON
        std::string json = R"({
    "model_cutoff": 4615159644819978768,
    "full_list": false,
    "type": "NeighborsListOptions"
})";
        options = NeighborsListOptionsHolder::from_json(json);
        CHECK(options->model_cutoff() == 3.5426);
        CHECK(options->full_list() == false);

        CHECK_THROWS_WITH(
            NeighborsListOptionsHolder::from_json("{}"),
            StartsWith("expected 'type' in JSON for NeighborsListOptions, did not find it")
        );
        CHECK_THROWS_WITH(
            NeighborsListOptionsHolder::from_json("{\"type\": \"nope\"}"),
            StartsWith("'type' in JSON for NeighborsListOptions must be 'NeighborsListOptions'")
        );
    }

    SECTION("ModelOutput") {
        // save to JSON
        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->quantity = "foo";
        output->unit = "bar";
        output->per_atom = false;
        output->forward_gradients = {"baz", "not.this-one_"};

        const auto* expected = R"({
    "forward_gradients": [
        "baz",
        "not.this-one_"
    ],
    "per_atom": false,
    "quantity": "foo",
    "type": "ModelOutput",
    "unit": "bar"
})";
        CHECK(output->to_json() == expected);

        // load from JSON
        std::string json = R"({
    "type": "ModelOutput",
    "quantity": "quantity",
    "forward_gradients": []
})";
        output = ModelOutputHolder::from_json(json);
        CHECK(output->quantity == "quantity");
        CHECK(output->unit.empty());
        CHECK(output->per_atom == false);
        CHECK(output->forward_gradients.empty());

        CHECK_THROWS_WITH(
            ModelOutputHolder::from_json("{}"),
            StartsWith("expected 'type' in JSON for ModelOutput, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelOutputHolder::from_json("{\"type\": \"nope\"}"),
            StartsWith("'type' in JSON for ModelOutput must be 'ModelOutput'")
        );
    }

    SECTION("ModelRunOptions") {
        // save to JSON
        auto options = torch::make_intrusive<ModelRunOptionsHolder>();
        options->length_unit = "mm";

        options->outputs.insert("output_1", torch::make_intrusive<ModelOutputHolder>());

        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->per_atom = true;
        output->unit = "something";
        options->outputs.insert("output_2", output);

        const auto* expected = R"({
    "length_unit": "mm",
    "outputs": {
        "output_1": {
            "forward_gradients": [],
            "per_atom": false,
            "quantity": "",
            "type": "ModelOutput",
            "unit": ""
        },
        "output_2": {
            "forward_gradients": [],
            "per_atom": true,
            "quantity": "",
            "type": "ModelOutput",
            "unit": "something"
        }
    },
    "selected_atoms": null,
    "type": "ModelRunOptions"
})";
        CHECK(options->to_json() == expected);


        // load from JSON
        std::string json =R"({
    "length_unit": "very large",
    "outputs": {
        "foo": {
            "forward_gradients": ["test"],
            "type": "ModelOutput"
        }
    },
    "selected_atoms": [1, 2, 3],
    "type": "ModelRunOptions"
})";

        options = ModelRunOptionsHolder::from_json(json);
        CHECK(options->length_unit == "very large");
        CHECK(options->selected_atoms == std::vector<int64_t>{1, 2, 3});

        output = options->outputs.at("foo");
        CHECK(output->quantity.empty());
        CHECK(output->unit.empty());
        CHECK(output->per_atom == false);
        CHECK(output->forward_gradients == std::vector<std::string>{"test"});

        CHECK_THROWS_WITH(
            ModelRunOptionsHolder::from_json("{}"),
            StartsWith("expected 'type' in JSON for ModelRunOptions, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelRunOptionsHolder::from_json("{\"type\": \"nope\"}"),
            StartsWith("'type' in JSON for ModelRunOptions must be 'ModelRunOptions'")
        );
    }

    SECTION("ModelCapabilities") {
        // save to JSON
        auto capabilities = torch::make_intrusive<ModelCapabilitiesHolder>();
        capabilities->length_unit = "µm";
        capabilities->species = {1, 2, -43};

        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->per_atom = true;
        output->quantity = "something";
        capabilities->outputs.insert("bar", output);

        const auto* expected = R"({
    "length_unit": "\u00b5m",
    "outputs": {
        "bar": {
            "forward_gradients": [],
            "per_atom": true,
            "quantity": "something",
            "type": "ModelOutput",
            "unit": ""
        }
    },
    "species": [
        1,
        2,
        -43
    ],
    "type": "ModelCapabilities"
})";
        CHECK(capabilities->to_json() == expected);


        // load from JSON
        std::string json =R"({
    "length_unit": "\u00b5m",
    "outputs": {
        "foo": {
            "forward_gradients": ["test"],
            "type": "ModelOutput"
        }
    },
    "species": [
        1,
        -2
    ],
    "type": "ModelCapabilities"
})";

        capabilities = ModelCapabilitiesHolder::from_json(json);
        CHECK(capabilities->length_unit == "µm");
        CHECK(capabilities->species == std::vector<int64_t>{1, -2});

        output = capabilities->outputs.at("foo");
        CHECK(output->quantity.empty());
        CHECK(output->unit.empty());
        CHECK(output->per_atom == false);
        CHECK(output->forward_gradients == std::vector<std::string>{"test"});

        CHECK_THROWS_WITH(
            ModelCapabilitiesHolder::from_json("{}"),
            StartsWith("expected 'type' in JSON for ModelCapabilities, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelCapabilitiesHolder::from_json("{\"type\": \"nope\"}"),
            StartsWith("'type' in JSON for ModelCapabilities must be 'ModelCapabilities'")
        );
    }
}