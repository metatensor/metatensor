#pragma once

#include <cmath>

#include <algorithm>
#include <functional>
#include <chrono>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <metatensor.h>

using Clock = std::chrono::steady_clock;

struct BenchmarkResult {
    double mean = 0.0;
    double std = 0.0;
    double min = 0.0;
    double max = 0.0;
};

const size_t MIN_ITERS = 20;
const size_t MAX_ITERS = 1000;
const double MAX_TOTAL_TIME_S = 3;

template <typename Function, typename Setup, typename TearDown>
inline BenchmarkResult bench_function_full(
    Function&& function,
    Setup&& setup,
    TearDown&& teardown,
    std::optional<std::size_t> n_iters,
    std::optional<std::size_t> n_warmup
) {
    for (std::size_t i = 0; i < n_warmup.value_or(5); i++) {
        auto args = setup();
        std::apply(function, std::move(args));
    }

    auto times = std::vector<double>();

    if (n_iters.has_value()) {
        times.reserve(*n_iters);
        for (std::size_t i = 0; i < *n_iters; i++) {
            auto args = setup();
            auto start = Clock::now();
            std::apply(function, std::move(args));
            auto end = Clock::now();
            auto elapsed = std::chrono::duration<double>(end - start).count();
            times.push_back(elapsed);
        }
    } else {
        double total_time = 0.0;
        std::size_t iters = 0;
        while (iters < MIN_ITERS || (total_time <= MAX_TOTAL_TIME_S && iters <= MAX_ITERS)) {
            auto start = Clock::now();
            auto args = setup();

            auto bench_start = Clock::now();
            std::apply(function, std::move(args));
            auto bench_end = Clock::now();
            auto bench_elapsed = std::chrono::duration<double>(bench_end - bench_start).count();

            times.push_back(bench_elapsed);

            auto end = Clock::now();
            total_time += std::chrono::duration<double>(end - start).count();
            iters += 1;
        }
    }

    auto minmax = std::minmax_element(times.begin(), times.end());
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());

    double variance = 0.0;
    for (auto t : times) {
        double diff = t - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(times.size());

    teardown();

    return BenchmarkResult{
        mean,
        std::sqrt(variance),
        *minmax.first,
        *minmax.second,
    };
}

template <typename Function, typename Setup>
inline BenchmarkResult bench_function_with_setup(
    Function&& function,
    Setup&& setup,
    std::optional<std::size_t> n_iters,
    std::optional<std::size_t> n_warmup
) {
    return bench_function_full(
        std::forward<Function>(function),
        std::forward<Setup>(setup),
        []() {},
        n_iters,
        n_warmup
    );
}

template <typename Function, typename TearDown>
inline BenchmarkResult bench_function_with_teardown(
    Function&& function,
    TearDown&& teardown,
    std::optional<std::size_t> n_iters,
    std::optional<std::size_t> n_warmup
) {
    return bench_function_full(
        std::forward<Function>(function),
        []() { return std::make_tuple(); },
        std::forward<TearDown>(teardown),
        n_iters,
        n_warmup
    );
}

template <typename Function>
inline BenchmarkResult bench_function(
    Function&& function,
    std::optional<std::size_t> n_iters,
    std::optional<std::size_t> n_warmup
) {
    return bench_function_full(
        std::forward<Function>(function),
        []() { return std::make_tuple(); },
        []() {},
        n_iters,
        n_warmup
    );
}

struct CommandLineOptions {
    bool json = false;
    bool list = false;
    bool test = false;
    std::vector<std::string> patterns;

    static CommandLineOptions parse(int argc, const char* argv[]);
};

using BenchFunction = std::function<BenchmarkResult(std::optional<std::size_t> n_iters, std::optional<std::size_t> n_warmup)>;

void run_benchmarks(
    const std::string& benchmark_name,
    const CommandLineOptions& options,
    const std::vector<std::pair<std::string, BenchFunction>>& benchmarks
);

mts_array_t make_empty_array(std::vector<uintptr_t> shape);

mts_array_t make_i32_array(
    std::shared_ptr<std::vector<int32_t>> data,
    std::vector<uintptr_t> shape
);
