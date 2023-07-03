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

struct UserData {
    std::string name;
    int64_t count;
    std::vector<double> values;
};

// some data will be stored here to check `user_data_delete` has been called
static int DELETE_CALL_MARKER = 0;

TEST_CASE("User data") {
    {
        // check that we can register and recover user data on Labels
        auto data = new UserData{"aabbccdd", 0xDEADBEEF, {1.0, 2.0, 3.0, 42.1}};
        auto user_data = equistore::LabelsUserData(data, [](void* ptr){
            DELETE_CALL_MARKER += 1;
            delete static_cast<UserData*>(ptr);
        });

        auto labels = Labels({"sample"}, {{0}, {1}, {2}});
        labels.set_user_data(std::move(user_data));

        auto data_ptr = static_cast<UserData*>(labels.user_data());
        REQUIRE(data_ptr != nullptr);
        CHECK(data_ptr->name == "aabbccdd");
        CHECK(data_ptr->count == 0xDEADBEEF);
        CHECK(data_ptr->values == std::vector<double>{1.0, 2.0, 3.0, 42.1});

        // check we can recover user data from Labels inside a TensorBlock
        auto block = TensorBlock(
            std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 2})),
            std::move(labels),
            {},
            Labels({"properties"}, {{5}, {3}})
        );

        auto samples = block.samples();
        data_ptr = static_cast<UserData*>(samples.user_data());
        REQUIRE(data_ptr != nullptr);
        CHECK(data_ptr->name == "aabbccdd");
        CHECK(data_ptr->count == 0xDEADBEEF);
        CHECK(data_ptr->values == std::vector<double>{1.0, 2.0, 3.0, 42.1});

        // check that we can mutate the data in-place
        data_ptr->count = 42;
        data_ptr->values.push_back(12356);

        samples = block.samples();
        data_ptr = static_cast<UserData*>(samples.user_data());
        REQUIRE(data_ptr != nullptr);
        CHECK(data_ptr->name == "aabbccdd");
        CHECK(data_ptr->count == 42);
        CHECK(data_ptr->values == std::vector<double>{1.0, 2.0, 3.0, 42.1, 12356});

        // check that we can register user data after the construction of the block
        data = new UserData{"properties", 0, {}};
        user_data = equistore::LabelsUserData(data, [](void* ptr){
            DELETE_CALL_MARKER += 10000000;
            delete static_cast<UserData*>(ptr);
        });

        auto properties = block.properties();
        properties.set_user_data(std::move(user_data));

        properties = block.properties();
        data_ptr = static_cast<UserData*>(properties.user_data());
        REQUIRE(data_ptr != nullptr);
        CHECK(data_ptr->name == "properties");
        CHECK(data_ptr->count == 0);
        CHECK(data_ptr->values == std::vector<double>{});
    }

    // Check that the `user_data_delete` function was called the
    // appropriate number of times
    CHECK(DELETE_CALL_MARKER == 10000001);
}
