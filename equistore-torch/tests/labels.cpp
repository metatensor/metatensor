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

    SECTION("print") {
        auto labels = LabelsHolder::create({"aaa", "bbb"}, {{1, 2}, {3, 4}});

        auto expected = " aaa  bbb\n     1    2\n     3    4\n";
        CHECK(labels->print(-1, 3) == expected);
    }

    SECTION("create views") {
        auto labels = LabelsHolder::create({"aaa", "bbb"}, {{1, 2}, {3, 4}});

        CHECK_FALSE(labels->is_view());

        auto view = LabelsHolder::view(labels, {"aaa"});
        CHECK(view->is_view());

        CHECK(view->names().size() == 1);
        CHECK(view->names()[0] == "aaa");

        CHECK(view->values().index({0, 0}).item<int32_t>() == 1);
        CHECK(view->values().index({1, 0}).item<int32_t>() == 3);

        CHECK_THROWS_WITH(view->position(1), Catch::Matchers::Contains("can not call this function on Labels view, call to_owned first"));
        auto owned = view->to_owned();
        CHECK_FALSE(owned.is_view());
        CHECK(owned.position(std::vector<int64_t>{1}).value() == 0);
    }

    SECTION("equality") {
        auto labels_1 = LabelsHolder::create({"a", "b"}, {{0, 0}, {1, 0}});
        auto labels_2 = LabelsHolder::create({"a", "b"}, {{0, 0}, {1, 0}});
        auto labels_3 = LabelsHolder::create({"a", "c"}, {{0, 0}, {1, 0}});

        // torch::intrusive_ptr operator== checks for object identity
        CHECK(labels_1 != labels_2);

        CHECK(*labels_1 == *labels_2);
        CHECK(*labels_1 != *labels_3);
    }
}

TEST_CASE("LabelsEntry") {

    auto labels = LabelsHolder::create({"aaa", "bbb"}, {{1, 2}, {3, 4}});

    auto entry = LabelsEntryHolder(labels, 0);
    CHECK(entry.size() == 2);

    CHECK(entry.names().size() == 2);
    CHECK(entry.names()[0] == "aaa");
    CHECK(entry.names()[1] == "bbb");

    CHECK(entry.values()[0].item<int32_t>() == 1);
    CHECK(entry.values()[1].item<int32_t>() == 2);

    CHECK(entry.__repr__() == "LabelsEntry(aaa=1, bbb=2)");

    CHECK(entry[0] == 1);
    CHECK(entry[1] == 2);

    CHECK(entry["aaa"] == 1);
    CHECK(entry["bbb"] == 2);
}
