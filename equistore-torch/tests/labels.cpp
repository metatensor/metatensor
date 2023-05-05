#include <torch/torch.h>

#include <equistore/torch.hpp>
using namespace equistore_torch;

#include <catch.hpp>

TEST_CASE("Labels") {
    SECTION("constructor") {
        torch::IValue names = std::vector<std::string>{"a", "bb"};
        auto values = std::vector<int64_t>{0, 0, 1, 0, 0, -1, 1, -2};
        auto labels = LabelsHolder(names, torch::tensor(values).reshape({4, 2}));

        CHECK(labels.count() == 4);
        CHECK(labels.size() == 2);

        // check the TorchScript version of the functions
        CHECK(labels.names().size() == 2);
        CHECK(labels.names()[0] == "a");
        CHECK(labels.names()[1] == "bb");

        CHECK((labels.values().sizes() == std::vector<int64_t>{4, 2}));
        CHECK(labels.values()[0][0].item<int64_t>() == 0);
        CHECK(labels.values()[3][1].item<int64_t>() == -2);

        // and the underlying equistore::Labels
        auto& equistore_labels = labels.as_equistore();
        CHECK(equistore_labels.names().size() == 2);
        CHECK(equistore_labels.names()[0] == std::string("a"));
        CHECK(equistore_labels.names()[1] == std::string("bb"));

        CHECK(equistore_labels(1, 1) == 0);
        CHECK(equistore_labels(3, 0) == 1);
    }

    SECTION("position") {
        auto labels = LabelsHolder::create({"a", "bb"}, {{0, 0}, {1, 0}, {0, 1}, {1, 1}});

        auto i = labels->position(std::vector<int64_t>{0, 1});
        CHECK(i.value() == 2);

        i = labels->position(std::vector<int64_t>{0, 4});
        CHECK_FALSE(i.has_value());
    }
}
