#include <iostream>

#include <catch.hpp>

#include <metatensor/torch.hpp>

#include "../src/internal/shared_libraries.cpp"


TEST_CASE("List shared libraries") {
    // force linking to metatensor_torch
    CHECK(!metatensor_torch::version().empty());

    auto libraries = metatensor_torch::details::get_loaded_libraries();

    bool found_metatensor = false;
    bool found_metatensor_torch = false;
    for (const auto& path: libraries) {
        if (path.find("metatensor_torch") != std::string::npos) {
            found_metatensor_torch = true;
            continue;
        }

        if (path.find("metatensor") != std::string::npos) {
            found_metatensor = true;
            continue;
        }
    }

    CHECK(found_metatensor);
    CHECK(found_metatensor_torch);
}
