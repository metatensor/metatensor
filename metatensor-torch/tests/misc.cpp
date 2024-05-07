#include <catch.hpp>

#include "metatensor/torch.hpp"


TEST_CASE("Version macros") {
    CHECK(std::string(METATENSOR_TORCH_VERSION) == metatensor_torch::version());

    auto version = std::to_string(METATENSOR_TORCH_VERSION_MAJOR) + "."
        + std::to_string(METATENSOR_TORCH_VERSION_MINOR) + "."
        + std::to_string(METATENSOR_TORCH_VERSION_PATCH);

    // METATENSOR_TORCH_VERSION should start with `x.y.z`
    CHECK(std::string(METATENSOR_TORCH_VERSION).find(version) == 0);
}
