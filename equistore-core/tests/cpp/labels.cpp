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
