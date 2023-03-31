#include <torch/torch.h>

#include <equistore/torch.hpp>
using namespace equistore_torch;

#include <catch.hpp>

TEST_CASE("Blocks") {
    SECTION("constructors and data") {
        auto block = TensorBlockHolder(
            torch::full({3, 2}, 11.0),
            LabelsHolder::create({"s"}, {{0}, {2}, {1}}),
            {},
            LabelsHolder::create({"p"}, {{0}, {1}})
        );

        CHECK((block.values().sizes() == std::vector<int64_t>{3, 2}));
        CHECK(torch::all(block.values() == 11.0).item<bool>());

        CHECK(*block.samples() == equistore::Labels({"s"}, {{0}, {2}, {1}}));
        CHECK(block.samples()->names().toTupleRef().size() == 1);
        CHECK(block.samples()->names().toTupleRef().elements()[0].toStringRef() == "s");

        CHECK(block.components().empty());

        CHECK(*block.properties() == equistore::Labels({"p"}, {{0}, {1}}));
        CHECK(block.properties()->names().toTupleRef().size() == 1);
        CHECK(block.properties()->names().toTupleRef().elements()[0].toStringRef() == "p");
    }

    SECTION("clone") {
        auto values = torch::full({3, 2}, 11.0);
        auto block = TensorBlockHolder(
            values,
            LabelsHolder::create({"s"}, {{0}, {2}, {1}}),
            {},
            LabelsHolder::create({"p"}, {{0}, {1}})
        );

        CHECK(values.data_ptr() == block.values().data_ptr());

        auto clone = block.copy();
        CHECK(values.data_ptr() != clone->values().data_ptr());
        CHECK(torch::all(values == clone->values()).item<bool>());
    }

    SECTION("gradients") {
        auto block = TensorBlockHolder(
            torch::full({3, 2}, 11.0),
            LabelsHolder::create({"s"}, {{0}, {2}, {1}}),
            {},
            LabelsHolder::create({"p"}, {{0}, {1}})
        );

        CHECK(block.gradients_list().empty());

        block.add_gradient("g", torch::make_intrusive<TensorBlockHolder>(
            torch::full({1, 3, 2}, 1.0),
            LabelsHolder::create({"sample", "g"}, {{0, 1}}),
            std::vector<TorchLabels>{LabelsHolder::create({"c"}, {{0}, {1}, {2}})},
            block.properties()
        ));

        CHECK((block.gradients_list() == std::vector<std::string>{"g"}));
        CHECK(block.has_gradient("g"));
        CHECK_FALSE(block.has_gradient("not-there"));

        auto gradient = block.gradient("g");
        CHECK((gradient->values().sizes() == std::vector<int64_t>{1, 3, 2}));

        auto sample_names = gradient->samples()->names().toTupleRef().elements();
        CHECK(sample_names.size() == 2);
        CHECK(sample_names[0].toStringRef() == "sample");
        CHECK(sample_names[1].toStringRef() == "g");

        auto gradients = block.gradients().toGenericDict();

        for (const auto& entry: gradients) {
            CHECK(entry.key().toStringRef() == "g");
        }
    }
}
