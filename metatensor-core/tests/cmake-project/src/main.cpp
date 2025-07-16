#include <metatensor.hpp>


int main() {
    try {
        auto labels = metatensor::Labels({"name"}, {{0}, {1}, {2}});
        auto labels_unchecked = metatensor::Labels({"name_uniq"}, {{0}, {1}, {2}}, metatensor::unchecked_t{});
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}
