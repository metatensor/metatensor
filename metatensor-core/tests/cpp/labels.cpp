#include <fstream>
#include <sstream>

#include <catch.hpp>

#include <metatensor.hpp>
using namespace metatensor;

TEST_CASE("Labels") {
    std::unique_ptr<DataArrayBase> array = std::make_unique<SimpleDataArray<int32_t>>(
        std::vector<size_t>{3, 2},
        std::vector<int32_t>{1, 2, 3, 4, 5, 6}
    );
    auto mts_array = DataArrayBase::to_mts_array(std::move(array));
    auto labels = Labels({"foo", "bar"}, std::move(mts_array));

    auto values = labels.values();
    CHECK(values.shape().size() == 2);
    CHECK(values.shape()[0] == 3);
    CHECK(values.shape()[1] == 2);

    CHECK(labels.count() == 3);
    CHECK(labels.size() == 2);

    CHECK(labels.names().size() == 2);
    CHECK(labels.names()[0] == std::string("foo"));
    CHECK(labels.names()[1] == std::string("bar"));

    CHECK(labels.position({3, 4}) == 1);
    CHECK(labels.position({1, 4}) == -1);

    CHECK(values(0, 0) == 1);
    CHECK(values(0, 1) == 2);

    CHECK(values(1, 0) == 3);
    CHECK(values(1, 1) == 4);

    CHECK(values(2, 0) == 5);
    CHECK(values(2, 1) == 6);

    const auto& values_cpu = labels.values_cpu();
    CHECK(values_cpu == values);

    CHECK(labels == Labels({"foo", "bar"}, {{1, 2}, {3, 4}, {5, 6}}));
    CHECK(labels != Labels({"bar", "foo"}, {{1, 2}, {3, 4}, {5, 6}}));
    CHECK(labels != Labels({"foo", "bar"}, {{1, 2}, {3, 4}, {5, 5}}));

    CHECK_THROWS_WITH(
        labels.position({3, 4, 5}),
        "invalid parameter: expected label of size 2 in mts_labels_position, got size 3"
    );

    CHECK_THROWS_WITH(Labels({"foo"}, {{1}, {3, 4}}), "invalid size for row: expected 1 got 2");

    CHECK_THROWS_WITH(
        Labels({"not an ident"}),
        "invalid parameter: 'not an ident' is not a valid label name"
    );

    auto empty = Labels(std::vector<std::string>{});
    CHECK(empty.size() == 0);
    CHECK(empty.count() == 0);
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
        const auto& values = union_.values();
        CHECK(values(0, 0) == 0);
        CHECK(values(0, 1) == 1);

        CHECK(values(1, 0) == 1);
        CHECK(values(1, 1) == 2);

        CHECK(values(2, 0) == 2);
        CHECK(values(2, 1) == 3);

        CHECK(values(3, 0) == 4);
        CHECK(values(3, 1) == 5);

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
        const auto& values = intersection.values();
        CHECK(values(0, 0) == 1);
        CHECK(values(0, 1) == 2);

        auto expected = std::vector<int64_t>{-1, 0};
        CHECK(first_mapping == expected);

        expected = std::vector<int64_t>{-1, 0, -1};
        CHECK(second_mapping == expected);
    }

    SECTION("difference") {
        auto first = Labels({"aa", "bb"}, {{0, 1}, {1, 2}});
        auto second = Labels({"aa", "bb"}, {{2, 3}, {1, 2}, {4, 5}});

        auto mapping = std::vector<int64_t>(first.count());

        auto difference = first.set_difference(second, mapping);
        CHECK(difference.size() == 2);
        CHECK(difference.names()[0] == std::string("aa"));
        CHECK(difference.names()[1] == std::string("bb"));

        CHECK(difference.count() == 1);
        auto values = difference.values();
        CHECK(values(0, 0) == 0);
        CHECK(values(0, 1) == 1);

        auto expected = std::vector<int64_t>{0, -1};
        CHECK(mapping == expected);

        mapping.resize(second.count());
        difference = second.set_difference(first, mapping);

        CHECK(difference.size() == 2);
        CHECK(difference.names()[0] == std::string("aa"));
        CHECK(difference.names()[1] == std::string("bb"));

        CHECK(difference.count() == 2);
        values = difference.values();
        CHECK(values(0, 0) == 2);
        CHECK(values(0, 1) == 3);

        CHECK(values(1, 0) == 4);
        CHECK(values(1, 1) == 5);

        expected = std::vector<int64_t>{0, -1, 1};
        CHECK(mapping == expected);
    }

    SECTION("select") {
        // selection with a subset of names
        auto labels = Labels({"aa", "bb"}, {{1, 1}, {1, 2}, {3, 2}, {2, 1}});
        auto selection = Labels({"aa"}, {{1}, {2}, {5}});

        auto selected = labels.select(selection);
        CHECK(selected.size() == 3);
        CHECK(selected == std::vector<int64_t>{0, 1, 3});

        // selection with the same names
        selection = Labels({"aa", "bb"}, {{1, 1}, {2, 1}, {5, 1}, {1, 2}});

        selected = labels.select(selection);
        CHECK(selected.size() == 3);
        CHECK(selected == std::vector<int64_t>{0, 3, 1});

        // empty selection
        selection = Labels({"aa"}, {});

        selected = labels.select(selection);
        CHECK(selected.size() == 0);

        // invalid selection names
        selection = Labels({"aaa"}, {{1}});
        CHECK_THROWS_WITH(labels.select(selection),
            "invalid parameter: 'aaa' in selection is not part of these Labels"
        );
    }
}

void check_loaded_labels(metatensor::Labels& labels) {
    CHECK(labels.names().size() == 4);
    CHECK(labels.names()[0] == std::string("o3_lambda"));
    CHECK(labels.names()[1] == std::string("o3_sigma"));
    CHECK(labels.names()[2] == std::string("center_type"));
    CHECK(labels.names()[3] == std::string("neighbor_type"));
    CHECK(labels.count() == 27);
}

TEST_CASE("Serialization") {
    SECTION("loading file") {
        // TEST_KEYS_NPY_PATH is defined by cmake and expand to the path of
        // `tests/keys.npy`
        auto labels = Labels::load(TEST_KEYS_MTS_PATH);
        check_loaded_labels(labels);

        labels = metatensor::io::load_labels(TEST_KEYS_MTS_PATH);
        check_loaded_labels(labels);

        CHECK_THROWS_WITH(
            Labels::load(TEST_BLOCK_MTS_PATH),
            Catch::Matchers::Contains("use `load_block` to load TensorBlock")
        );
        CHECK_THROWS_WITH(
            metatensor::io::load_labels(TEST_BLOCK_MTS_PATH),
            Catch::Matchers::Contains("use `load_block` to load TensorBlock")
        );

        CHECK_THROWS_WITH(
            Labels::load(TEST_DATA_MTS_PATH),
            Catch::Matchers::Contains("use `load` to load TensorMap")
        );
        CHECK_THROWS_WITH(
            metatensor::io::load_labels(TEST_DATA_MTS_PATH),
            Catch::Matchers::Contains("use `load` to load TensorMap")
        );
    }

    SECTION("Load/Save with buffers") {
        // read the whole file into a buffer
        std::ifstream file(TEST_KEYS_MTS_PATH, std::ios::binary);
        std::ostringstream string_stream;
        string_stream << file.rdbuf();
        auto buffer = string_stream.str();

        auto labels = Labels::load_buffer(buffer);
        check_loaded_labels(labels);

        labels = metatensor::io::load_labels_buffer(buffer);
        check_loaded_labels(labels);

        auto saved = labels.save_buffer<std::string>();
        REQUIRE(saved.size() == buffer.size());
        CHECK(saved == buffer);

        saved = metatensor::io::save_buffer<std::string>(labels);
        REQUIRE(saved.size() == buffer.size());
        CHECK(saved == buffer);

        // using the raw C API, making the callback a small wrapper around
        // std::realloc
        uint8_t* raw_buffer = nullptr;
        uintptr_t buflen = 0;
        auto status = mts_labels_save_buffer(
            &raw_buffer,
            &buflen,
            nullptr,
            [](void*, uint8_t* ptr, uintptr_t new_size){
                return static_cast<uint8_t*>(std::realloc(ptr, new_size));
            },
            labels.as_mts_labels_t()
        );
        REQUIRE(status == MTS_SUCCESS);
        CHECK(saved == std::string(raw_buffer, raw_buffer + buflen));

        std::free(raw_buffer);
    }
}
