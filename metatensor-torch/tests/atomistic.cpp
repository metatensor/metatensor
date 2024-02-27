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

    SECTION("ModelMetadata") {
        // save to JSON
        auto metadata = torch::make_intrusive<ModelMetadataHolder>();
        metadata->name = "some name";
        metadata->description = "describing it";
        metadata->authors = {"John Doe", "Napoleon"};
        metadata->references.insert("model", std::vector<std::string>{"some-ref"});
        metadata->references.insert("architecture", std::vector<std::string>{"ref-2", "ref-3"});

        const auto* expected = R"({
    "authors": [
        "John Doe",
        "Napoleon"
    ],
    "class": "ModelMetadata",
    "description": "describing it",
    "name": "some name",
    "references": {
        "architecture": [
            "ref-2",
            "ref-3"
        ],
        "model": [
            "some-ref"
        ]
    }
})";
        CHECK(metadata->to_json() == expected);


// Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

        // load from JSON
        std::string json =R"({
    "class": "ModelMetadata",
    "name": "foo",
    "description": "test",
    "authors": ["me", "myself"],
    "references": {
        "implementation": ["torch-power!"],
        "model": ["took a while to train"]
    }
})";

        metadata = ModelMetadataHolder::from_json(json);
        CHECK(metadata->name == "foo");
        CHECK(metadata->description == "test");
        CHECK(metadata->authors == std::vector<std::string>{"me", "myself"});
        CHECK(metadata->references.at("implementation") == std::vector<std::string>{"torch-power!"});
        CHECK(metadata->references.at("model") == std::vector<std::string>{"took a while to train"});

        CHECK_THROWS_WITH(
            ModelMetadataHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelMetadata, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelMetadataHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelMetadata must be 'ModelMetadata'")
        );

        // printing
        metadata = torch::make_intrusive<ModelMetadataHolder>();
        metadata->name = "name";
        metadata->description = R"(Lorem ipsum dolor sit amet, consectetur
adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna
aliqua. Ut enim ad minim veniam, quis nostrud exercitation.)";
        metadata->authors = {"Short author", "Some extremely long author that will take more than one line in the printed output"};
        metadata->references.insert("model", std::vector<std::string>{
            "a very long reference that will take more than one line in the printed output"
        });
        metadata->references.insert("architecture", std::vector<std::string>{"ref-2", "ref-3"});

        expected = R"(This is the name model
======================

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation.

Model authors
-------------

- Short author
- Some extremely long author that will take more than one line in the printed
  output

Model references
----------------

Please cite the following references when using this model:
- about this specific model:
  * a very long reference that will take more than one line in the printed
    output
- about the architecture of this model:
  * ref-2
  * ref-3
)";

        CHECK(metadata->print() == expected);
    }
}
