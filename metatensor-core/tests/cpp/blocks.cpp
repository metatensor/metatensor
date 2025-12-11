#include <fstream>
#include <sstream>

#include <catch.hpp>

#include <metatensor.hpp>
using namespace metatensor;

static void check_loaded_block(metatensor::TensorBlock& block);

TEST_CASE("Blocks") {
    SECTION("no components") {
        auto block = TensorBlock(
            std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({3, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            {},
            Labels({"properties"}, {{5}, {3}})
        );

        auto values_mts_array = block.mts_array();
        CHECK(SimpleDataArray<double>::from_mts_array(values_mts_array).shape() == std::vector<size_t>{3, 2});

        auto values = block.values();
        CHECK(values.shape() == std::vector<size_t>{3, 2});

        CHECK(block.samples() == Labels({"samples"}, {{0}, {1}, {4}}));
        CHECK(block.components().empty());
        CHECK(block.properties() == Labels({"properties"}, {{5}, {3}}));
    }

    SECTION("with components") {
        auto components = std::vector<Labels>();
        components.emplace_back(Labels({"component_1"}, {{-1}, {0}, {1}}));
        components.emplace_back(Labels({"component_2"}, {{-4}, {1}}));
        auto block = TensorBlock(
            std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({3, 3, 2, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            components,
            Labels({"properties"}, {{5}, {3}})
        );

        auto values_mts_array = block.mts_array();
        CHECK(SimpleDataArray<double>::from_mts_array(values_mts_array).shape() == std::vector<size_t>{3, 3, 2, 2});

        auto values = block.values();
        CHECK(values.shape() == std::vector<size_t>{3, 3, 2, 2});

        CHECK(block.samples() == Labels({"samples"}, {{0}, {1}, {4}}));

        components = block.components();
        CHECK(components.size() == 2);
        CHECK(components[0] == Labels({"component_1"}, {{-1}, {0}, {1}}));
        CHECK(components[1] == Labels({"component_2"}, {{-4}, {1}}));

        CHECK(block.properties() == Labels({"properties"}, {{5}, {3}}));
    }

    SECTION("gradients") {
        auto components = std::vector<Labels>();
        components.emplace_back(Labels({"component"}, {{-1}, {0}, {1}}));
        auto properties = Labels({"properties"}, {{5}, {3}});
        auto block = TensorBlock(
            std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({3, 3, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            components,
            properties
        );

        components = std::vector<Labels>();
        components.emplace_back(Labels({"gradient_component"}, {{42}}));
        components.emplace_back(Labels({"component"}, {{-1}, {0}, {1}}));

        auto gradient = TensorBlock(
            std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({2, 1, 3, 2})),
            Labels({"sample", "parameter"}, {{0, -2}, {2, 3}}),
            components,
            properties
        );
        block.add_gradient("parameter", std::move(gradient));

        CHECK(block.gradients_list() == std::vector<std::string>{"parameter"});

        auto gradient_mts_array = block.gradient("parameter").mts_array();
        CHECK(SimpleDataArray<double>::from_mts_array(gradient_mts_array).shape() == std::vector<size_t>{2, 1, 3, 2});

        gradient = block.gradient("parameter");
        gradient_mts_array = gradient.mts_array();
        CHECK(SimpleDataArray<double>::from_mts_array(gradient_mts_array).shape() == std::vector<size_t>{2, 1, 3, 2});
        auto data = gradient.values();
        CHECK(data.shape() == std::vector<size_t>{2, 1, 3, 2});

        CHECK(gradient.samples() == Labels({"sample", "parameter"}, {{0, -2}, {2, 3}}));

        components = gradient.components();
        CHECK(components.size() == 2);
        CHECK(components[0] == Labels({"gradient_component"}, {{42}}));
        CHECK(components[1] == Labels({"component"}, {{-1}, {0}, {1}}));

        CHECK(gradient.properties() == Labels({"properties"}, {{5}, {3}}));

        CHECK_THROWS_WITH(
            block.gradient("not there"),
            "invalid parameter: can not find gradients with respect to 'not there' in this block"
        );
    }

    SECTION("clone") {
        auto block = TensorBlock(
            std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({3, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            {},
            Labels({"properties"}, {{5}, {3}})
        );

        // This should be fine
        auto clone = block.clone();


        class BrokenDataArray: public metatensor::SimpleDataArray<double> {
        public:
            BrokenDataArray(std::vector<size_t> shape): metatensor::SimpleDataArray<double>(std::move(shape)) {}

            std::unique_ptr<DataArrayBase> copy() const override {
                throw std::runtime_error("can not copy this!");
            }
        };

        block = TensorBlock(
            std::unique_ptr<BrokenDataArray>(new BrokenDataArray({3, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            {},
            Labels({"properties"}, {{5}, {3}})
        );

        CHECK_THROWS_WITH(block.clone(), "external error: calling mts_array_t.create failed (status -1)");
    }


    SECTION("clone metadata") {
        auto components = std::vector<Labels>();
        components.emplace_back(Labels({"component_1"}, {{-1}, {0}, {1}}));
        components.emplace_back(Labels({"component_2"}, {{-4}, {1}}));
        auto block = TensorBlock(
            std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({3, 3, 2, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            components,
            Labels({"properties"}, {{5}, {3}})
        );

        auto clone = block.clone_metadata_only();

        CHECK(clone.samples() == block.samples());
        CHECK(clone.components() == block.components());
        CHECK(clone.properties() == block.properties());

        auto array = clone.mts_array();

        const uintptr_t* shape = nullptr;
        uintptr_t shape_count = 0;
        CHECK(array.shape(array.ptr, &shape, &shape_count) == MTS_SUCCESS);
        CHECK(std::vector<uintptr_t>(shape, shape + shape_count) == std::vector<uintptr_t>{{3, 3, 2, 2}});

        mts_data_origin_t origin = 0;
        CHECK(array.origin(array.ptr, &origin) == MTS_SUCCESS);

        char buffer[32];
        CHECK(mts_get_data_origin(origin, buffer, sizeof(buffer)) == MTS_SUCCESS);
        CHECK(buffer == std::string("metatensor::EmptyDataArray"));
    }

    SECTION("empty labels") {
        auto block = TensorBlock(
            std::unique_ptr<SimpleDataArray<double>>(new SimpleDataArray<double>({3, 0})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            {},
            Labels({}, {})
        );
    }
}

TEST_CASE("Serialization") {
    SECTION("loading file") {
        // TEST_BLOCK_MTS_PATH is defined by cmake and expand to the path of
        // `tests/block.mts`
        auto block = TensorBlock::load(TEST_BLOCK_MTS_PATH);
        check_loaded_block(block);

        block = metatensor::io::load_block(TEST_BLOCK_MTS_PATH);
        check_loaded_block(block);

        CHECK_THROWS_WITH(
            TensorBlock::load(TEST_KEYS_MTS_PATH),
            Catch::Matchers::Contains("use `load_labels` to load Labels")
        );
        CHECK_THROWS_WITH(
            metatensor::io::load_block(TEST_KEYS_MTS_PATH),
            Catch::Matchers::Contains("use `load_labels` to load Labels")
        );

        CHECK_THROWS_WITH(
            TensorBlock::load(TEST_DATA_MTS_PATH),
            Catch::Matchers::Contains("use `load` to load TensorMap")
        );
        CHECK_THROWS_WITH(
            metatensor::io::load_block(TEST_DATA_MTS_PATH),
            Catch::Matchers::Contains("use `load` to load TensorMap")
        );
    }

    SECTION("Load/Save with buffers") {
        // read the whole file into a buffer
        std::ifstream file(TEST_BLOCK_MTS_PATH, std::ios::binary);
        std::ostringstream string_stream;
        string_stream << file.rdbuf();
        auto buffer = string_stream.str();

        auto block = TensorBlock::load_buffer(buffer);
        check_loaded_block(block);

        block = metatensor::io::load_block_buffer(buffer);
        check_loaded_block(block);

        auto saved = block.save_buffer<std::string>();
        REQUIRE(saved.size() == buffer.size());
        CHECK(saved == buffer);

        saved = metatensor::io::save_buffer<std::string>(block);
        REQUIRE(saved.size() == buffer.size());
        CHECK(saved == buffer);

        // using the raw C API, making the callback a small wrapper around
        // std::realloc
        uint8_t* raw_buffer = nullptr;
        uintptr_t buflen = 0;
        auto status = mts_block_save_buffer(
            &raw_buffer,
            &buflen,
            nullptr,
            [](void*, uint8_t* ptr, uintptr_t new_size){
                return static_cast<uint8_t*>(std::realloc(ptr, new_size));
            },
            block.as_mts_block_t()
        );
        REQUIRE(status == MTS_SUCCESS);
        CHECK(saved == std::string(raw_buffer, raw_buffer + buflen));

        std::free(raw_buffer);
    }
}

void check_loaded_block(metatensor::TensorBlock& block) {
    auto samples = block.samples();
    CHECK(samples.names().size() == 2);
    CHECK(samples.names()[0] == std::string("system"));
    CHECK(samples.names()[1] == std::string("atom"));

    auto values = block.values();
    CHECK(values.shape() == std::vector<size_t>{9, 5, 3});

    auto gradient = block.gradient("positions");
    samples = gradient.samples();
    CHECK(samples.names().size() == 3);
    CHECK(samples.names()[0] == std::string("sample"));
    CHECK(samples.names()[1] == std::string("system"));
    CHECK(samples.names()[2] == std::string("atom"));

    values = gradient.values();
    CHECK(values.shape() == std::vector<size_t>{59, 3, 5, 3});
}
