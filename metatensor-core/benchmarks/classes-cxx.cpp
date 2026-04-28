#include <cstdint>

#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <metatensor.hpp>

#include "./bench_utils.hpp"


metatensor::Labels range_labels(const std::string& name, int32_t count) {
    auto values = std::vector<int32_t>(static_cast<std::size_t>(count));
    std::iota(values.begin(), values.end(), int32_t{0});
    return metatensor::Labels({name}, values.data(), values.size());
}

BenchmarkResult bench_labels_small(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    return bench_function(
        []() {
            auto labels = metatensor::Labels(
                {"a", "b", "c"},
                {
                    {0, 0, 0},
                    {0, 0, 1},
                    {0, 1, 0},
                    {1, 0, 0},
                    {1, 1, 1},
                }
            );
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_labels_large(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    auto values = std::vector<int32_t>();
    values.reserve(100 * 100 * 100 * 3);
    for (int32_t i = 0; i < 100; i++) {
        for (int32_t j = 0; j < 100; j++) {
            for (int32_t k = 0; k < 100; k++) {
                values.push_back(i);
                values.push_back(j);
                values.push_back(k);
            }
        }
    }

    return bench_function(
        [values = std::move(values)]() {
            auto labels = metatensor::Labels({"a", "b", "c"}, values.data(), values.size() / 3);
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_labels_large_assume_unique(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    auto values = std::vector<int32_t>();
    values.reserve(100 * 100 * 100 * 3);
    for (int32_t i = 0; i < 100; i++) {
        for (int32_t j = 0; j < 100; j++) {
            for (int32_t k = 0; k < 100; k++) {
                values.push_back(i);
                values.push_back(j);
                values.push_back(k);
            }
        }
    }

    return bench_function(
        [values = std::move(values)]() {
            auto labels = metatensor::Labels(
                {"a", "b", "c"},
                values.data(),
                values.size() / 3,
                metatensor::assume_unique{}
            );
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_tensor_block(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    auto samples = range_labels("samples", 100);
    auto properties = range_labels("properties", 100);
    auto values = std::vector<double>(100 * 100);

    return bench_function(
        [samples, properties, values = std::move(values)]() {
            auto block = metatensor::TensorBlock(
                std::make_unique<metatensor::EmptyDataArray>(
                    std::vector<uintptr_t>{100, 100}
                ),
                samples,
                {},
                properties
            );
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_tensor_block_large(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    auto samples = range_labels("samples", 10000);
    auto components = std::vector<metatensor::Labels>{
        range_labels("component_1", 10),
        range_labels("component_2", 5),
        range_labels("component_3", 10),
    };
    auto properties = range_labels("properties", 100);

    return bench_function(
        [samples, components, properties]() {
            auto block = metatensor::TensorBlock(
                std::make_unique<metatensor::EmptyDataArray>(
                    std::vector<uintptr_t>{10000, 10, 5, 10, 100}
                ),
                samples,
                components,
                properties
            );
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_tensor_map(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    constexpr int32_t n_blocks = 10;

    auto prepare_blocks = []() {
        auto blocks = std::vector<metatensor::TensorBlock>();
        blocks.reserve(static_cast<std::size_t>(n_blocks));

        auto samples = range_labels("samples", 100);
        auto properties = range_labels("properties", 100);

        for (int32_t i = 0; i < n_blocks; i++) {
            blocks.emplace_back(
                std::make_unique<metatensor::EmptyDataArray>(
                    std::vector<uintptr_t>{100, 100}
                ),
                samples,
                std::vector<metatensor::Labels>{},
                properties
            );
        }

        return std::make_tuple(std::move(blocks));
    };

    auto keys_values = std::vector<int32_t>(static_cast<std::size_t>(n_blocks));
    std::iota(keys_values.begin(), keys_values.end(), int32_t{0});
    auto keys = metatensor::Labels({"key"}, keys_values.data(), keys_values.size());

    return bench_function_with_setup(
        [keys](std::vector<metatensor::TensorBlock> blocks) {
            auto tensor = metatensor::TensorMap(keys, std::move(blocks));
        },
        std::move(prepare_blocks),
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_tensor_map_large(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    constexpr int32_t n_blocks = 10000;

    auto prepare_blocks = []() {
        auto blocks = std::vector<metatensor::TensorBlock>();
        blocks.reserve(static_cast<std::size_t>(n_blocks));

        auto samples = range_labels("samples", 100);
        auto properties = range_labels("properties", 100);

        for (int32_t i = 0; i < n_blocks; i++) {
            blocks.emplace_back(
                std::make_unique<metatensor::EmptyDataArray>(
                    std::vector<uintptr_t>{100, 100}
                ),
                samples,
                std::vector<metatensor::Labels>{},
                properties
            );
        }

        return std::make_tuple(std::move(blocks));
    };

    auto keys_values = std::vector<int32_t>(static_cast<std::size_t>(n_blocks));
    std::iota(keys_values.begin(), keys_values.end(), int32_t{0});
    auto keys = metatensor::Labels({"key"}, keys_values.data(), keys_values.size());

    return bench_function_with_setup(
        [keys](std::vector<metatensor::TensorBlock> blocks) {
            auto tensor = metatensor::TensorMap(keys, std::move(blocks));
        },
        std::move(prepare_blocks),
        n_iters,
        n_warmup
    );
}

int main(int argc, const char* argv[]) {
    const auto benchmarks = std::vector<std::pair<std::string, BenchFunction>>{
        {"Labels/small", bench_labels_small},
        {"Labels/large", bench_labels_large},
        {"Labels/large_assume_unique", bench_labels_large_assume_unique},
        {"TensorBlock/small", bench_tensor_block},
        {"TensorBlock/large", bench_tensor_block_large},
        {"TensorMap/small", bench_tensor_map},
        {"TensorMap/large", bench_tensor_map_large},
    };

    auto options = CommandLineOptions::parse(argc, argv);

    run_benchmarks("C++ API", options, benchmarks);
}
