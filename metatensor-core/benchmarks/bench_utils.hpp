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
    size_t samples = 0;
    double mean = 0.0;
    double std = 0.0;
    double min = 0.0;
    double max = 0.0;
};

const size_t MIN_SAMPLES = 20;
const double MAX_TOTAL_TIME_S = 2.0;

// when a single call is faster than this threshold, we will batch multiple
// calls together to get more accurate timing (default: 1µs)
const double BENCHMARK_BATCH_THRESHOLD_S = 1e-6;
const size_t BATCH_SIZE = 100;

std::vector<double> remove_outliers(std::vector<double> times);

template <typename Function, typename Setup, typename TearDown>
inline BenchmarkResult bench_function_full(
    Function&& function,
    Setup&& setup,
    TearDown&& teardown,
    std::optional<std::size_t> n_samples,
    std::optional<std::size_t> n_warmup
) {
    for (std::size_t i = 0; i < n_warmup.value_or(5); i++) {
        auto args = setup();
        std::apply(function, std::move(args));
    }

    bool use_batching = false;
    {
        auto args = setup();
        auto start = Clock::now();
        std::apply(function, std::move(args));
        auto end = Clock::now();
        auto elapsed = std::chrono::duration<double>(end - start).count();
        if (elapsed < BENCHMARK_BATCH_THRESHOLD_S) {
            use_batching = true;
        }
    }

    // check if batching is needed by timing one extra call

    auto times = std::vector<double>();
    if (use_batching) {
        if (n_samples.has_value()) {
            times.reserve(*n_samples);
            for (std::size_t i = 0; i < *n_samples; i++) {
                auto all_args = std::vector<decltype(setup())>();
                all_args.reserve(BATCH_SIZE);
                for (std::size_t j = 0; j < BATCH_SIZE; j++) {
                    all_args.push_back(setup());
                }
                auto start = Clock::now();
                for (auto& args : all_args) {
                    std::apply(function, std::move(args));
                }
                auto end = Clock::now();
                auto elapsed = std::chrono::duration<double>(end - start).count() / static_cast<double>(BATCH_SIZE);
                times.push_back(elapsed);
            }
        } else {
            double total_time = 0.0;
            std::size_t n = 0;
            while (n < MIN_SAMPLES || total_time <= MAX_TOTAL_TIME_S) {
                auto all_args = std::vector<decltype(setup())>();
                all_args.reserve(BATCH_SIZE);
                for (std::size_t j = 0; j < BATCH_SIZE; j++) {
                    all_args.push_back(setup());
                }
                auto start = Clock::now();
                for (auto& args : all_args) {
                    std::apply(function, std::move(args));
                }
                auto end = Clock::now();
                auto elapsed = std::chrono::duration<double>(end - start).count() / static_cast<double>(BATCH_SIZE);
                times.push_back(elapsed);
                total_time += std::chrono::duration<double>(end - start).count();
                n += 1;
            }
        }

        times = remove_outliers(std::move(times));
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
            times.size(),
            mean,
            std::sqrt(variance),
            *minmax.first,
            *minmax.second,
        };
    }

    // normal (non-batched) timing
    if (n_samples.has_value()) {
        times.reserve(*n_samples);
        for (std::size_t i = 0; i < *n_samples; i++) {
            auto args = setup();
            auto start = Clock::now();
            std::apply(function, std::move(args));
            auto end = Clock::now();
            auto elapsed = std::chrono::duration<double>(end - start).count();
            times.push_back(elapsed);
        }
    } else {
        double total_time = 0.0;
        std::size_t n = 0;
        while (n < MIN_SAMPLES || total_time <= MAX_TOTAL_TIME_S) {
            auto start = Clock::now();
            auto args = setup();

            auto bench_start = Clock::now();
            std::apply(function, std::move(args));
            auto bench_end = Clock::now();
            auto bench_elapsed = std::chrono::duration<double>(bench_end - bench_start).count();

            times.push_back(bench_elapsed);

            auto end = Clock::now();
            total_time += std::chrono::duration<double>(end - start).count();
            n += 1;
        }
    }

    times = remove_outliers(std::move(times));
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
        times.size(),
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
    std::optional<std::size_t> n_samples,
    std::optional<std::size_t> n_warmup
) {
    return bench_function_full(
        std::forward<Function>(function),
        std::forward<Setup>(setup),
        []() {},
        n_samples,
        n_warmup
    );
}

template <typename Function, typename TearDown>
inline BenchmarkResult bench_function_with_teardown(
    Function&& function,
    TearDown&& teardown,
    std::optional<std::size_t> n_samples,
    std::optional<std::size_t> n_warmup
) {
    return bench_function_full(
        std::forward<Function>(function),
        []() { return std::make_tuple(); },
        std::forward<TearDown>(teardown),
        n_samples,
        n_warmup
    );
}

template <typename Function>
inline BenchmarkResult bench_function(
    Function&& function,
    std::optional<std::size_t> n_samples,
    std::optional<std::size_t> n_warmup
) {
    return bench_function_full(
        std::forward<Function>(function),
        []() { return std::make_tuple(); },
        []() {},
        n_samples,
        n_warmup
    );
}

struct CommandLineOptions {
    bool json = false;
    bool list = false;
    bool test = false;
    bool help = false;
    std::string save_baseline;
    std::string baseline;
    std::string binary_path;
    std::vector<std::string> patterns;

    static CommandLineOptions parse(int argc, const char* argv[]);
};

using BenchFunction = std::function<BenchmarkResult(std::optional<std::size_t> n_samples, std::optional<std::size_t> n_warmup)>;

void run_benchmarks(
    const std::string& benchmark_name,
    const std::string& baseline_prefix,
    const CommandLineOptions& options,
    const std::vector<std::pair<std::string, BenchFunction>>& benchmarks
);

mts_array_t make_empty_array(std::vector<uintptr_t> shape);

mts_array_t make_i32_array(
    std::shared_ptr<std::vector<int32_t>> data,
    std::vector<uintptr_t> shape
);
