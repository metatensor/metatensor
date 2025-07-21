#include <metatensor.hpp>


int main() {
    try {
        auto labels = metatensor::Labels({"name"}, {{0}, {1}, {2}});
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}
