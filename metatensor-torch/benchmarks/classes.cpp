#include <cstdint>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "metatensor/torch/block.hpp"
#include "metatensor/torch/labels.hpp"
#include "metatensor/torch/tensor.hpp"

#include "./bench_utils.hpp"

using metatensor_torch::Labels;
using metatensor_torch::LabelsHolder;
using metatensor_torch::TensorBlock;
using metatensor_torch::TensorBlockHolder;
using metatensor_torch::TensorMapHolder;

static torch::TensorOptions i32_cpu() {
    return torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
}

static torch::TensorOptions f32_cpu() {
    return torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
}

static Labels range_labels(const std::string& name, int32_t count) {
    auto values = torch::arange(static_cast<int64_t>(count), i32_cpu()).reshape({count, 1});
    return torch::make_intrusive<LabelsHolder>(name, std::move(values));
}

BenchmarkResult bench_labels_small(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    return bench_function(
        []() {
            torch::make_intrusive<LabelsHolder>(
                std::vector<std::string>{"a", "b", "c"},
                torch::tensor(
                    {
                        {0, 0, 0},
                        {0, 0, 1},
                        {0, 1, 0},
                        {1, 0, 0},
                        {1, 1, 1},
                    },
                    i32_cpu()
                )
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

    auto tensor = torch::tensor(values, i32_cpu()).reshape({100 * 100 * 100, 3});
    return bench_function(
        [tensor]() {
            torch::make_intrusive<LabelsHolder>(
                std::vector<std::string>{"a", "b", "c"},
                tensor
            );
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

    auto tensor = torch::tensor(values, i32_cpu()).reshape({100 * 100 * 100, 3});
    return bench_function(
        [tensor]() {
            torch::make_intrusive<LabelsHolder>(
                std::vector<std::string>{"a", "b", "c"},
                tensor,
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
    auto values = torch::rand({100, 100}, f32_cpu());

    return bench_function(
        [samples, properties, values]() {
            torch::make_intrusive<TensorBlockHolder>(
                values,
                samples,
                std::vector<Labels>{},
                properties
            );
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_tensor_block_large(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    auto samples = range_labels("samples", 10000);
    auto components = std::vector<Labels>{
        range_labels("component_1", 10),
        range_labels("component_2", 5),
        range_labels("component_3", 10),
    };
    auto properties = range_labels("properties", 100);
    auto values = torch::rand({10000, 10, 5, 10, 100}, f32_cpu());

    return bench_function(
        [samples, components, properties, values]() {
            torch::make_intrusive<TensorBlockHolder>(
                values,
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
        auto blocks = std::vector<TensorBlock>();
        blocks.reserve(static_cast<std::size_t>(n_blocks));

        auto samples = range_labels("samples", 100);
        auto properties = range_labels("properties", 100);
        auto values = torch::rand({100, 100}, f32_cpu());

        for (int32_t i = 0; i < n_blocks; i++) {
            blocks.emplace_back(torch::make_intrusive<TensorBlockHolder>(
                values,
                samples,
                std::vector<Labels>{},
                properties
            ));
        }

        return std::make_tuple(std::move(blocks));
    };

    auto keys = range_labels("key", n_blocks);

    return bench_function_with_setup(
        [keys](std::vector<TensorBlock> blocks) {
            torch::make_intrusive<TensorMapHolder>(keys, std::move(blocks));
        },
        std::move(prepare_blocks),
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_tensor_map_large(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    constexpr int32_t n_blocks = 10000;

    auto prepare_blocks = []() {
        auto blocks = std::vector<TensorBlock>();
        blocks.reserve(static_cast<std::size_t>(n_blocks));

        auto samples = range_labels("samples", 100);
        auto properties = range_labels("properties", 100);
        auto values = torch::rand({100, 100}, f32_cpu());

        for (int32_t i = 0; i < n_blocks; i++) {
            blocks.emplace_back(torch::make_intrusive<TensorBlockHolder>(
                values,
                samples,
                std::vector<Labels>{},
                properties
            ));
        }

        return std::make_tuple(std::move(blocks));
    };

    auto keys = range_labels("key", n_blocks);

    return bench_function_with_setup(
        [keys](std::vector<TensorBlock> blocks) {
            torch::make_intrusive<TensorMapHolder>(keys, std::move(blocks));
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
    run_benchmarks("Torch C++ API", options, benchmarks);
}
