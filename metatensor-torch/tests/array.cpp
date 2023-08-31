#include <torch/torch.h>

#include <metatensor/torch.hpp>
using namespace metatensor_torch;

#include <catch.hpp>

TEST_CASE("Arrays") {
    auto tensor = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kF64));
    auto array = TorchDataArray(tensor);

    SECTION("origin") {
        auto origin = array.origin();

        char buffer[64] = {0};
        auto status = mts_get_data_origin(origin, buffer, 64);
        CHECK(status == MTS_SUCCESS);
        CHECK(std::string(buffer) == "metatensor_torch::TorchDataArray");
    }

    SECTION("shape") {
        auto shape = array.shape();
        CHECK(shape.size() == 3);
        CHECK(shape[0] == 2);
        CHECK(shape[1] == 3);
        CHECK(shape[2] == 4);

        CHECK((array.tensor().sizes() == std::vector<int64_t>{2, 3, 4}));

        array.reshape({1, 2, 3, 4});
        shape = array.shape();
        CHECK(shape.size() == 4);
        CHECK(shape[0] == 1);
        CHECK(shape[1] == 2);
        CHECK(shape[2] == 3);
        CHECK(shape[3] == 4);

        CHECK((array.tensor().sizes() == std::vector<int64_t>{1, 2, 3, 4}));

        array.swap_axes(1, 2);
        shape = array.shape();
        CHECK(shape.size() == 4);
        CHECK(shape[0] == 1);
        CHECK(shape[1] == 3);
        CHECK(shape[2] == 2);
        CHECK(shape[3] == 4);

        CHECK((array.tensor().sizes() == std::vector<int64_t>{1, 3, 2, 4}));
    }

    SECTION("new arrays") {
        auto copy = array.copy();
        auto copy_ptr = dynamic_cast<TorchDataArray*>(copy.get());

        CHECK(copy_ptr->tensor().data_ptr() != array.tensor().data_ptr());
        CHECK((copy_ptr->tensor().sizes() == std::vector<int64_t>{2, 3, 4}));
        CHECK(copy_ptr->tensor().dtype() == torch::kF64);

        auto created = array.create({5, 6});
        auto created_ptr = dynamic_cast<TorchDataArray*>(created.get());

        CHECK((created_ptr->tensor().sizes() == std::vector<int64_t>{5, 6}));
        CHECK(created_ptr->tensor().dtype() == torch::kF64);
    }
}
