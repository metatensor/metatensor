#include <catch.hpp>

#include <equistore.hpp>
using namespace equistore;

TEST_CASE("Labels") {
    auto labels = Labels({"foo", "bar"}, {{1, 2}, {3, 4}, {5, 6}});

    CHECK(labels.shape().size() == 2);
    CHECK(labels.shape()[0] == 3);
    CHECK(labels.shape()[1] == 2);

    CHECK(labels.names().size() == 2);
    CHECK(labels.names()[0] == std::string("foo"));
    CHECK(labels.names()[1] == std::string("bar"));

    CHECK(labels.position({3, 4}) == 1);
    CHECK(labels.position({1, 4}) == -1);

    CHECK(labels(0, 0) == 1);
    CHECK(labels(0, 1) == 2);

    CHECK(labels(1, 0) == 3);
    CHECK(labels(1, 1) == 4);

    CHECK(labels(2, 0) == 5);
    CHECK(labels(2, 1) == 6);

    CHECK(labels == Labels({"foo", "bar"}, {{1, 2}, {3, 4}, {5, 6}}));
    CHECK(labels != Labels({"bar", "foo"}, {{1, 2}, {3, 4}, {5, 6}}));
    CHECK(labels != Labels({"foo", "bar"}, {{1, 2}, {3, 4}, {5, 5}}));

    CHECK_THROWS_WITH(
        labels.position({3, 4, 5}),
        "invalid parameter: expected label of size 2 in eqs_labels_position, got size 3"
    );

    CHECK_THROWS_WITH(Labels({"foo"}, {{1}, {3, 4}}), "invalid size for row: expected 1 got 2");

    CHECK_THROWS_WITH(
        Labels({"not an ident"}, {{0}}),
        "invalid parameter: 'not an ident' is not a valid label name"
    );
}


TEST_CASE("Set operations") {
    SECTION("union") {
        auto first = Labels({"aa", "bb"}, {{0, 1}, {1, 2}});
        auto second = Labels({"aa", "bb"}, {{2, 3}, {1, 2}, {4, 5}});

        auto first_mapping = std::vector<int64_t>(first.count());
        auto second_mapping = std::vector<int64_t>(second.count());

        auto union_ = first.set_union(second, first_mapping, second_mapping);

        CHECK(union_.size() == 2);
        CHECK(union_.names()[0] == std::string("aa"));
        CHECK(union_.names()[1] == std::string("bb"));

        CHECK(union_.count() == 4);
        CHECK(union_(0, 0) == 0);
        CHECK(union_(0, 1) == 1);

        CHECK(union_(1, 0) == 1);
        CHECK(union_(1, 1) == 2);

        CHECK(union_(2, 0) == 2);
        CHECK(union_(2, 1) == 3);

        CHECK(union_(3, 0) == 4);
        CHECK(union_(3, 1) == 5);

        auto expected = std::vector<int64_t>{0, 1};
        CHECK(first_mapping == expected);

        expected = std::vector<int64_t>{2, 1, 3};
        CHECK(second_mapping == expected);
    }

    SECTION("intersection") {
        auto first = Labels({"aa", "bb"}, {{0, 1}, {1, 2}});
        auto second = Labels({"aa", "bb"}, {{2, 3}, {1, 2}, {4, 5}});

        auto first_mapping = std::vector<int64_t>(first.count());
        auto second_mapping = std::vector<int64_t>(second.count());

        auto intersection = first.set_intersection(second, first_mapping, second_mapping);

        CHECK(intersection.size() == 2);
        CHECK(intersection.names()[0] == std::string("aa"));
        CHECK(intersection.names()[1] == std::string("bb"));

        CHECK(intersection.count() == 1);
        CHECK(intersection(0, 0) == 1);
        CHECK(intersection(0, 1) == 2);

        auto expected = std::vector<int64_t>{-1, 0};
        CHECK(first_mapping == expected);

        expected = std::vector<int64_t>{-1, 0, -1};
        CHECK(second_mapping == expected);
    }
}
