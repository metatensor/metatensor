#include <catch.hpp>

#include "metatensor.h"


TEST_CASE("Version macros") {
    CHECK(std::string(METATENSOR_VERSION) == mts_version());

    auto version = std::to_string(METATENSOR_VERSION_MAJOR) + "."
        + std::to_string(METATENSOR_VERSION_MINOR) + "."
        + std::to_string(METATENSOR_VERSION_PATCH);

    // METATENSOR_VERSION should start with `x.y.z`
    CHECK(std::string(METATENSOR_VERSION).find(version) == 0);
}
