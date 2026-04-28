#include <cstdint>

#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <metatensor.h>

#include "./bench_utils.hpp"


std::string last_error_message() {
    const char* message = nullptr;
    const char* origin = nullptr;
    void* data = nullptr;
    auto status = mts_last_error(&message, &origin, &data);
    (void)data;

    if (status != MTS_SUCCESS || message == nullptr) {
        return "unknown metatensor error";
    }

    if (origin == nullptr) {
        return std::string(message);
    }

    return std::string(origin) + ": " + message;
}

[[noreturn]] void throw_last_error(const std::string& context) {
    throw std::runtime_error(context + ": " + last_error_message());
}

template <typename Pointer>
Pointer* check_pointer(Pointer* ptr, const std::string& context) {
    if (ptr == nullptr) {
        throw_last_error(context);
    }
    return ptr;
}


const mts_labels_t* range_labels(const char* name, uintptr_t count) {
    auto values = std::make_shared<std::vector<int32_t>>(static_cast<size_t>(count));
    std::iota(values->begin(), values->end(), int32_t{0});

    auto shape = std::vector<uintptr_t>{count, 1};
    auto array = make_i32_array(std::move(values), std::move(shape));

    const char* names[] = {name};
    const auto* labels = mts_labels_assume_unique(names, 1, array);
    check_pointer(labels, "mts_labels_assume_unique");

    return labels;
}


BenchmarkResult bench_labels_small(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    const char* names[] = {"a", "b", "c"};
    auto values = std::make_shared<std::vector<int32_t>>(
        std::initializer_list<int32_t>{
            0, 0, 0,
            0, 0, 1,
            0, 1, 0,
            1, 0, 0,
            1, 1, 1,
        }
    );

    return bench_function_with_setup(
        [names](mts_array_t array) {
            const auto* labels = mts_labels(names, 3, array);
            check_pointer(labels, "mts_labels");
            mts_labels_free(labels);
        },
        [values]() {
            return std::make_tuple(make_i32_array(values, {5, 3}));
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_labels_large(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    auto values = std::make_shared<std::vector<int32_t>>();
    values->reserve(100 * 100 * 100 * 3);
    for (int32_t i = 0; i < 100; i++) {
        for (int32_t j = 0; j < 100; j++) {
            for (int32_t k = 0; k < 100; k++) {
                values->push_back(i);
                values->push_back(j);
                values->push_back(k);
            }
        }
    }

    const char* names[] = {"a", "b", "c"};
    return bench_function_with_setup(
        [names](mts_array_t array) {
            const auto* labels = mts_labels(names, 3, array);
            check_pointer(labels, "mts_labels");
            mts_labels_free(labels);
        },
        [values]() {
            return std::make_tuple(make_i32_array(values, {100 * 100 * 100, 3}));
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_labels_large_assume_unique(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    auto values = std::make_shared<std::vector<int32_t>>();
    values->reserve(100 * 100 * 100 * 3);
    for (int32_t i = 0; i < 100; i++) {
        for (int32_t j = 0; j < 100; j++) {
            for (int32_t k = 0; k < 100; k++) {
                values->push_back(i);
                values->push_back(j);
                values->push_back(k);
            }
        }
    }

    const char* names[] = {"a", "b", "c"};
    return bench_function_with_setup(
        [names](mts_array_t array) {
            const auto* labels = mts_labels_assume_unique(names, 3, array);
            check_pointer(labels, "mts_labels_assume_unique");
            mts_labels_free(labels);
        },
        [values]() {
            return std::make_tuple(make_i32_array(values, {100 * 100 * 100, 3}));
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_tensor_block(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    const auto* samples = range_labels("samples", 100);
    const auto* properties = range_labels("properties", 100);

    return bench_function_with_teardown(
        [samples, properties]() {
            auto* block = mts_block(
                make_empty_array({100, 100}),
                samples,
                nullptr,
                0,
                properties
            );
            check_pointer(block, "mts_block");
            mts_block_free(block);
        },
        [=]() {
            mts_labels_free(samples);
            mts_labels_free(properties);
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_tensor_block_large(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    const auto* samples = range_labels("samples", 10000);
    const auto* component_1 = range_labels("component_1", 10);
    const auto* component_2 = range_labels("component_2", 5);
    const auto* component_3 = range_labels("component_3", 10);
    const auto* properties = range_labels("properties", 100);

    const mts_labels_t* components[3] = {component_1, component_2, component_3};

    return bench_function_with_teardown(
        [=]() {
            auto* block = mts_block(
                make_empty_array({10000, 10, 5, 10, 100}),
                samples,
                components,
                3,
                properties
            );
            check_pointer(block, "mts_block");
            mts_block_free(block);
        },
        [=]() {
            mts_labels_free(samples);
            mts_labels_free(component_1);
            mts_labels_free(component_2);
            mts_labels_free(component_3);
            mts_labels_free(properties);
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_tensor_map(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    size_t n_blocks = 10;

    const auto* keys = range_labels("key", n_blocks);

    auto prepare_blocks = [n_blocks]() {
        auto blocks = std::vector<mts_block_t*>();
        const auto* samples = range_labels("samples", 100);
        const auto* properties = range_labels("properties", 100);

        for (size_t i = 0; i < n_blocks; i++) {
            auto* block = check_pointer(
                mts_block(
                    make_empty_array({100, 100}),
                    samples,
                    nullptr,
                    0,
                    properties
                ),
                "mts_block"
            );

            blocks.push_back(block);
        }

        mts_labels_free(samples);
        mts_labels_free(properties);

        return std::make_tuple(std::move(blocks));
    };

    return bench_function_full(
        [keys](std::vector<mts_block_t*> blocks) {
            auto* tensor = check_pointer(
                mts_tensormap(keys, blocks.data(), blocks.size()),
                "mts_tensormap"
            );

            mts_tensormap_free(tensor);
        },
        std::move(prepare_blocks),
        [=]() {
            mts_labels_free(keys);
        },
        n_iters,
        n_warmup
    );
}

BenchmarkResult bench_tensor_map_large(std::optional<size_t> n_iters, std::optional<size_t> n_warmup) {
    size_t n_blocks = 10000;

    const auto* keys = range_labels("key", n_blocks);

    auto prepare_blocks = [n_blocks]() {
        auto blocks = std::vector<mts_block_t*>();
        const auto* samples = range_labels("samples", 100);
        const auto* properties = range_labels("properties", 100);

        for (size_t i = 0; i < n_blocks; i++) {
            auto* block = check_pointer(
                mts_block(
                    make_empty_array({100, 100}),
                    samples,
                    nullptr,
                    0,
                    properties
                ),
                "mts_block"
            );

            blocks.push_back(block);
        }

        mts_labels_free(samples);
        mts_labels_free(properties);

        return std::make_tuple(std::move(blocks));
    };

    return bench_function_full(
        [keys](std::vector<mts_block_t*> blocks) {
            auto* tensor = check_pointer(
                mts_tensormap(keys, blocks.data(), blocks.size()),
                "mts_tensormap"
            );

            mts_tensormap_free(tensor);
        },
        std::move(prepare_blocks),
        [=]() {
            mts_labels_free(keys);
        },
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

    run_benchmarks("C API", options, benchmarks);
}
