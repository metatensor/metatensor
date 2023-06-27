#include <equistore/torch.hpp>

using namespace equistore_torch;

int main() {
    auto labels = LabelsHolder::create({"aa", "bb"}, {{0, 1}, {1, 1}, {2, 2}});

    return 0;
}
