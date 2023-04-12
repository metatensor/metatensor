#include <catch.hpp>

#include <equistore.hpp>
using namespace equistore;

TEST_CASE("Blocks") {
    SECTION("no components") {
        auto block = TensorBlock(
            std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            {},
            Labels({"properties"}, {{5}, {3}})
        );

        auto values_eqs_array = block.eqs_array();
        CHECK(SimpleDataArray::from_eqs_array(values_eqs_array).shape() == std::vector<size_t>{3, 2});

        auto values = block.values();
        CHECK(values.shape() == std::vector<size_t>{3, 2});

        CHECK(block.samples() == Labels({"samples"}, {{0}, {1}, {4}}));
        CHECK(block.components().size() == 0);
        CHECK(block.properties() == Labels({"properties"}, {{5}, {3}}));
    }

    SECTION("with components") {
        auto components = std::vector<Labels>();
        components.emplace_back(Labels({"component_1"}, {{-1}, {0}, {1}}));
        components.emplace_back(Labels({"component_2"}, {{-4}, {1}}));
        auto block = TensorBlock(
            std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 3, 2, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            std::move(components),
            Labels({"properties"}, {{5}, {3}})
        );

        auto values_eqs_array = block.eqs_array();
        CHECK(SimpleDataArray::from_eqs_array(values_eqs_array).shape() == std::vector<size_t>{3, 3, 2, 2});

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
            std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 3, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            components,
            properties
        );

        components = std::vector<Labels>();
        components.emplace_back(Labels({"gradient_component"}, {{42}}));
        components.emplace_back(Labels({"component"}, {{-1}, {0}, {1}}));

        auto gradient = TensorBlock(
            std::unique_ptr<SimpleDataArray>(new SimpleDataArray({2, 1, 3, 2})),
            Labels({"sample", "parameter"}, {{0, -2}, {2, 3}}),
            components,
            properties
        );
        block.add_gradient("parameter", std::move(gradient));

        CHECK(block.gradients_list() == std::vector<std::string>{"parameter"});

        auto gradient_eqs_array = block.gradient("parameter").eqs_array();
        CHECK(SimpleDataArray::from_eqs_array(gradient_eqs_array).shape() == std::vector<size_t>{2, 1, 3, 2});

        gradient = block.gradient("parameter");
        gradient_eqs_array = gradient.eqs_array();
        CHECK(SimpleDataArray::from_eqs_array(gradient_eqs_array).shape() == std::vector<size_t>{2, 1, 3, 2});
        auto data = gradient.values();
        CHECK(data.shape() == std::vector<size_t>{2, 1, 3, 2});

        CHECK(gradient.samples() == Labels({"sample", "parameter"}, {{0, -2}, {2, 3}}));

        components = gradient.components();
        CHECK(components.size() == 2);
        CHECK(components[0] == Labels({"gradient_component"}, {{42}}));
        CHECK(components[1] == Labels({"component"}, {{-1}, {0}, {1}}));

        CHECK(gradient.properties() == Labels({"properties"}, {{5}, {3}}));

        CHECK_THROWS_WITH(
            block.gradient("not there").values(),
            "invalid parameter: can not find gradients with respect to 'not there' in this block"
        );
    }

    SECTION("clone") {
        auto block = TensorBlock(
            std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            {},
            Labels({"properties"}, {{5}, {3}})
        );

        // This should be fine
        auto clone = block.clone();


        class BrokenDataArray: public equistore::SimpleDataArray {
        public:
            BrokenDataArray(std::vector<size_t> shape): equistore::SimpleDataArray(shape) {}

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

        CHECK_THROWS_WITH(block.clone(), "external error: calling eqs_array_t.create failed (status -1)");
    }
}
