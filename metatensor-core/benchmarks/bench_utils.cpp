#include <cassert>
#include <cstring>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./bench_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


std::string format_samples(size_t count) {
    if (count >= 1000000) {
        return std::to_string(count / 1000000) + "M";
    } else if (count >= 1000) {
        return std::to_string(count / 1000) + "k";
    } else {
        return std::to_string(count);
    }
}

std::pair<std::string, double> guess_unit(double time) {
    if (time < 1e-6) {
        return {"ns", 1e9};
    } else if (time < 1e-3) {
        return {"us", 1e6};
    } else if (time < 1) {
        return {"ms", 1e3};
    } else {
        return {"s", 1};
    }
}

std::string format_result_table(
    const std::vector<std::pair<std::string, BenchmarkResult>>& results,
    std::optional<std::unordered_map<std::string, BenchmarkResult>> baseline = std::nullopt
) {
    auto out = std::ostringstream();
    out << '\n';

    constexpr size_t name_width = 35;
    out << std::left << std::setw(name_width) << "Function"
        << std::right << std::setw(9) << "Samples"
        << std::setw(15) << "Mean"
        << std::setw(15) << "Std"
        << std::setw(15) << "Min"
        << std::setw(15) << "Max";
    if (baseline.has_value()) {
        out << std::setw(14) << "vs baseline";
    }
    out << '\n';

    size_t total_width = name_width + 69 + (baseline.has_value() ? 14 : 0);
    out << std::string(total_width, '-') << '\n';

    for (const auto& [name, result] : results) {
        auto [unit, scale] = guess_unit(result.mean);
        out << std::left << std::setw(name_width) << name
            << "  " << std::right << std::setw(7) << format_samples(result.samples)
            << "     " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << result.mean * scale << unit
            << "     " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << result.std * scale << unit
            << "     " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << result.min * scale << unit
            << "     " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << result.max * scale << unit;

        if (baseline.has_value()) {
            auto it = baseline->find(name);
            if (it != baseline->end()) {
                double ratio = result.mean / it->second.mean;
                out << "     " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << ratio << "x";
            } else {
                out << "           N/A";
            }
        }
        out << '\n';
    }

    return out.str();
}

std::string escape_json(const std::string& value) {
    auto escaped = std::string();
    escaped.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped.push_back(c); break;
        }
    }
    return escaped;
}

std::string format_result_json(
    const std::vector<std::pair<std::string, BenchmarkResult>>& results,
    std::optional<std::unordered_map<std::string, BenchmarkResult>> baseline = std::nullopt
) {
    auto out = std::ostringstream();
    out << "{";
    bool first = true;
    for (const auto& [name, result] : results) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << escape_json(name) << "\":{"
            << "\"samples\":" << result.samples << ","
            << "\"mean\":" << result.mean << ","
            << "\"std\":" << result.std << ","
            << "\"min\":" << result.min << ","
            << "\"max\":" << result.max;

        if (baseline.has_value()) {
            auto it = baseline->find(name);
            if (it != baseline->end()) {
                double ratio = result.mean / it->second.mean;
                out << ",\"vs_baseline\":" << ratio;
            }
        }
        out << "}";
    }
    out << "}";
    return out.str();
}

std::vector<double> remove_outliers(std::vector<double> times) {
    if (times.size() < 4) {
        return times;
    }

    auto sorted = times;
    std::sort(sorted.begin(), sorted.end());
    double q1 = sorted[sorted.size() / 4];
    double q3 = sorted[sorted.size() * 3 / 4];
    double iqr = q3 - q1;
    double lower = q1 - 3.0 * iqr;
    double upper = q3 + 3.0 * iqr;

    std::vector<double> filtered;
    std::copy_if(
        times.begin(), times.end(), std::back_inserter(filtered),
        [&](double t) { return t >= lower && t <= upper; }
    );

    auto original_size = times.size();
    if (filtered.size() * 10 < original_size * 5) {
        throw std::runtime_error(
            "error: more than 50% of samples were outliers ("
            + std::to_string(original_size - filtered.size()) + " out of " + std::to_string(original_size) + ")"
        );
    }

    if (filtered.size() < 1000 && original_size >= 1000) {
        throw std::runtime_error(
            "error: too many outliers, not enough samples left ("
            + std::to_string(filtered.size()) + " out of " + std::to_string(original_size) + ")"
        );
    }

    return filtered;
}

static void print_benchmark_usage(const std::string& program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS] [REGEX...]\n";
    std::cout << "Run benchmarks and print the results.\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --list                     list all available benchmarks and exit\n";
    std::cout << "  --json                     output the results as JSON instead of a table\n";
    std::cout << "  --test                     run in test mode (no warmup, 1 iteration)\n";
    std::cout << "  --save-baseline=NAME       save benchmark results as a baseline JSON file\n";
    std::cout << "  --save-baseline NAME       same as above\n";
    std::cout << "  --baseline=NAME            load baseline JSON and display speedup/slowdown\n";
    std::cout << "  --baseline NAME            same as above\n";
    std::cout << "  --help                     show this help message and exit\n";
    std::cout << "  REGEX                      filter benchmarks to run (default: .*)\n";
}

CommandLineOptions CommandLineOptions::parse(int argc, const char* argv[]) {
    auto options = CommandLineOptions();
    options.binary_path = argv[0];

    for (int i = 1; i < argc; i++) {
        auto arg = std::string(argv[i]);

        if (arg == "--json") {
            options.json = true;
        } else if (arg == "--list") {
            options.list = true;
        } else if (arg == "--test") {
            options.test = true;
        } else if (arg == "--help") {
            options.help = true;
        } else if (arg == "--save-baseline" && i + 1 < argc) {
            options.save_baseline = argv[++i];
        } else if (arg.rfind("--save-baseline=", 0) == 0) {
            options.save_baseline = arg.substr(std::strlen("--save-baseline="));
        } else if (arg == "--baseline" && i + 1 < argc) {
            options.baseline = argv[++i];
        } else if (arg.rfind("--baseline=", 0) == 0) {
            options.baseline = arg.substr(std::strlen("--baseline="));
        } else {
            options.patterns.emplace_back(std::move(arg));
        }
    }

    if (options.patterns.empty()) {
        options.patterns.emplace_back(".*");
    }

    return options;
}

// very minimal JSON parser, only supports the specific format we output in format_result_json
static std::unordered_map<std::string, BenchmarkResult> parse_baseline_json(const std::string& content) {
    std::unordered_map<std::string, BenchmarkResult> result;

    auto pos = content.find('{');
    if (pos == std::string::npos) {
        return result;
    }
    pos++;

    auto skip_whitespace = [&]() {
        while (pos < content.size() && (
            content[pos] == ' ' || content[pos] == '\n' ||
            content[pos] == '\t' || content[pos] == '\r')
        ) {
            pos++;
        }
    };

    auto parse_string = [&]() -> std::string {
        skip_whitespace();
        if (pos >= content.size() || content[pos] != '"') {
            return "";
        }
        pos++;
        std::string str;
        while (pos < content.size() && content[pos] != '"') {
            str += content[pos];
            pos++;
        }
        if (pos < content.size()) {
            pos++;
        }
        return str;
    };

    auto parse_number = [&]() -> double {
        skip_whitespace();
        size_t start = pos;
        if (pos < content.size() && (content[pos] == '-' || content[pos] == '+')) {
            pos++;
        }

        while (pos < content.size() && (std::isdigit(content[pos]) || content[pos] == '.')) {
            pos++;
        }

        if (pos < content.size() && (content[pos] == 'e' || content[pos] == 'E')) {
            pos++;

            if (pos < content.size() && (content[pos] == '-' || content[pos] == '+')) {
                pos++;
            }

            while (pos < content.size() && std::isdigit(content[pos])) {
                pos++;
            }
        }
        return std::stod(content.substr(start, pos - start));
    };

    while (pos < content.size()) {
        skip_whitespace();
        if (pos >= content.size() || content[pos] == '}') {
            break;
        }

        if (content[pos] == ',') {
            pos++; continue;
        }

        auto key = parse_string();
        if (key.empty()) {
            break;
        }

        skip_whitespace();
        if (pos >= content.size() || content[pos] != ':') {
            break;
        }
        pos++;

        skip_whitespace();
        if (pos >= content.size() || content[pos] != '{') {
            break;
        }
        pos++;

        BenchmarkResult br;
        while (pos < content.size()) {
            skip_whitespace();

            if (pos >= content.size() || content[pos] == '}') {
                break;
            }

            if (content[pos] == ',') {
                pos++;
                continue;
            }

            auto field = parse_string();
            skip_whitespace();

            if (pos >= content.size() || content[pos] != ':') {
                break;
            }
            pos++;

            double value = parse_number();
            if (field == "mean") {
                br.mean = value;
            } else if (field == "std") {
                br.std = value;
            } else if (field == "min") {
                br.min = value;
            } else if (field == "max") {
                br.max = value;
            }
        }

        if (pos < content.size() && content[pos] == '}') {
            pos++;
        }

        result[key] = br;
    }

    return result;
}

static std::filesystem::path baseline_directory(const std::string& binary_path) {
    auto* env = std::getenv("METATENSOR_BENCHMARK_BASELINE_DIR");
    if (env != nullptr) {
        return std::filesystem::path(env);
    }
    return std::filesystem::path(binary_path).parent_path();
}

void run_benchmarks(
    const std::string& benchmark_name,
    const std::string& baseline_prefix,
    const CommandLineOptions& options,
    const std::vector<std::pair<std::string, BenchFunction>>& benchmarks
) {
    if (options.help) {
        auto basename = std::filesystem::path(options.binary_path).filename().string();
        print_benchmark_usage(basename);
        return;
    }

    if (options.list) {
        std::cout << "Available benchmarks:\n";
        for (const auto& [name, _] : benchmarks) {
            (void)_;
            std::cout << " - " << name << "\n";
        }
        return;
    }

    auto compiled_patterns = std::vector<std::regex>();
    compiled_patterns.reserve(options.patterns.size());
    for (const auto& pattern : options.patterns) {
        compiled_patterns.emplace_back(pattern);
    }

    std::optional<size_t> n_samples = options.test ? std::make_optional<size_t>(1) : std::nullopt;
    std::optional<size_t> n_warmup = options.test ? std::make_optional<size_t>(0) : std::nullopt;

    auto results = std::vector<std::pair<std::string, BenchmarkResult>>();
    for (const auto& [name, function] : benchmarks) {
        bool should_run = false;
        for (const auto& pattern : compiled_patterns) {
            if (std::regex_search(name, pattern)) {
                should_run = true;
                break;
            }
        }

        if (!should_run) {
            continue;
        }

        try {
            std::cerr << "running " << name << "...\n";
            results.emplace_back(name, function(n_samples, n_warmup));
        } catch (const std::exception& error) {
            std::cerr << "error while running " << name << ": " << error.what() << "\n";
        }
    }

    auto baseline_dir = baseline_directory(options.binary_path);
    std::filesystem::create_directories(baseline_dir);

    if (!options.save_baseline.empty()) {
        auto baseline_name = baseline_prefix + "-" + options.save_baseline + ".json";
        auto baseline_path = baseline_dir / baseline_name;
        auto out = std::ofstream(baseline_path);
        out << format_result_json(results) << "\n";
    }

    std::optional<std::unordered_map<std::string, BenchmarkResult>> baseline_opt = std::nullopt;
    if (!options.baseline.empty()) {
        auto baseline_name = baseline_prefix + "-" + options.baseline + ".json";
        auto baseline_path = baseline_dir / baseline_name;
        auto in_file = std::ifstream(baseline_path);
        if (in_file) {
            auto content = std::string(
                std::istreambuf_iterator<char>(in_file),
                std::istreambuf_iterator<char>()
            );
            baseline_opt = parse_baseline_json(content);
        } else {
            std::cerr << "error: could not open baseline file: " << baseline_path << "\n";
        }
    }

    if (options.json) {
        std::cout << format_result_json(results, baseline_opt) << "\n";
    } else {
        std::cout << "Benchmark results for " << benchmark_name << ":\n";
        std::cout << format_result_table(results, baseline_opt);
    }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


static const char* BENCH_ARRAY_ORIGIN = "metatensor.bench.classes-c";
static std::once_flag BENCH_ARRAY_ORIGIN_ONCE;
static mts_data_origin_t BENCH_ARRAY_ORIGIN_ID = {};

struct BenchArray {
    bool is_i32 = false;
    std::vector<uintptr_t> shape;
    std::shared_ptr<std::vector<int32_t>> i32_data;
};

struct I32DLPackContext {
    std::shared_ptr<std::vector<int32_t>> data;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};

static void set_callback_error(const char* message) {
    mts_set_last_error(message, BENCH_ARRAY_ORIGIN, nullptr, nullptr);
}

static size_t product(const std::vector<uintptr_t>& shape) {
    size_t result = 1;
    for (auto value : shape) {
        result *= static_cast<size_t>(value);
    }
    return result;
}

static void dlpack_deleter(DLManagedTensorVersioned* managed) {
    if (managed == nullptr) {
        return;
    }

    auto* context = static_cast<I32DLPackContext*>(managed->manager_ctx);
    delete context;
    delete managed;
}

static void bench_array_destroy(void* array) {
    delete static_cast<BenchArray*>(array);
}

static mts_status_t bench_array_origin(const void*, mts_data_origin_t* origin) {
    std::call_once(BENCH_ARRAY_ORIGIN_ONCE, []() {
        auto status = mts_register_data_origin(BENCH_ARRAY_ORIGIN, &BENCH_ARRAY_ORIGIN_ID);
        assert(status == MTS_SUCCESS);
    });

    *origin = BENCH_ARRAY_ORIGIN_ID;
    return MTS_SUCCESS;
}

static mts_status_t bench_array_device(const void*, DLDevice* device) {
    device->device_type = kDLCPU;
    device->device_id = 0;
    return MTS_SUCCESS;
}

static mts_status_t bench_array_dtype(const void* array, DLDataType* dtype) {
    const auto* bench = static_cast<const BenchArray*>(array);

    if (bench->is_i32) {
        dtype->code = kDLInt;
        dtype->bits = 32;
        dtype->lanes = 1;
    } else {
        dtype->code = kDLFloat;
        dtype->bits = 64;
        dtype->lanes = 1;
    }

    return MTS_SUCCESS;
}

static mts_status_t bench_array_as_dlpack(
    void* array,
    DLManagedTensorVersioned** dl_managed_tensor,
    DLDevice device,
    const int64_t* stream,
    DLPackVersion
) {
    try {
        auto* bench = static_cast<BenchArray*>(array);

        if (!bench->is_i32) {
            set_callback_error("as_dlpack is not implemented for empty benchmark arrays");
            return MTS_CALLBACK_ERROR;
        }

        if (device.device_type != kDLCPU || device.device_id != 0) {
            set_callback_error("benchmark arrays only support CPU device");
            return MTS_CALLBACK_ERROR;
        }

        if (stream != nullptr) {
            set_callback_error("benchmark arrays do not support custom streams");
            return MTS_CALLBACK_ERROR;
        }

        auto* context = new I32DLPackContext();
        context->data = bench->i32_data;
        context->shape.reserve(bench->shape.size());
        for (auto value : bench->shape) {
            context->shape.push_back(static_cast<int64_t>(value));
        }

        context->strides.resize(context->shape.size());
        if (!context->shape.empty()) {
            context->strides.back() = 1;
            for (size_t i = context->shape.size() - 1; i > 0; i--) {
                context->strides[i - 1] = context->strides[i] * context->shape[i];
            }
        }

        auto* managed = new DLManagedTensorVersioned();
        managed->version.major = DLPACK_MAJOR_VERSION;
        managed->version.minor = DLPACK_MINOR_VERSION;
        managed->manager_ctx = context;
        managed->deleter = dlpack_deleter;
        managed->flags = 0;

        managed->dl_tensor.data = context->data->data();
        managed->dl_tensor.device = DLDevice{kDLCPU, 0};
        managed->dl_tensor.ndim = static_cast<int32_t>(context->shape.size());
        managed->dl_tensor.dtype = DLDataType{kDLInt, 32, 1};
        managed->dl_tensor.byte_offset = 0;

        if (context->shape.empty()) {
            managed->dl_tensor.shape = nullptr;
            managed->dl_tensor.strides = nullptr;
        } else {
            managed->dl_tensor.shape = context->shape.data();
            managed->dl_tensor.strides = context->strides.data();
        }

        *dl_managed_tensor = managed;
        return MTS_SUCCESS;
    } catch (const std::exception& error) {
        set_callback_error(error.what());
        return MTS_CALLBACK_ERROR;
    } catch (...) {
        set_callback_error("unexpected exception in benchmark array callback");
        return MTS_CALLBACK_ERROR;
    }
}

static mts_status_t bench_array_shape(const void* array, const uintptr_t** shape, uintptr_t* shape_count) {
    const auto* bench = static_cast<const BenchArray*>(array);
    *shape_count = static_cast<uintptr_t>(bench->shape.size());
    if (bench->shape.empty()) {
        *shape = nullptr;
    } else {
        *shape = bench->shape.data();
    }
    return MTS_SUCCESS;
}

static mts_status_t bench_array_reshape(void* array, const uintptr_t* shape, uintptr_t shape_count) {
    auto* bench = static_cast<BenchArray*>(array);
    auto new_shape = std::vector<uintptr_t>(shape, shape + shape_count);

    if (product(bench->shape) != product(new_shape)) {
        set_callback_error("invalid shape in reshape callback");
        return MTS_CALLBACK_ERROR;
    }

    bench->shape = std::move(new_shape);
    return MTS_SUCCESS;
}

static mts_status_t bench_array_swap_axes(void* array, uintptr_t axis_1, uintptr_t axis_2) {
    auto* bench = static_cast<BenchArray*>(array);
    if (axis_1 >= bench->shape.size() || axis_2 >= bench->shape.size()) {
        set_callback_error("axis out of range in swap_axes callback");
        return MTS_CALLBACK_ERROR;
    }

    std::swap(bench->shape[axis_1], bench->shape[axis_2]);
    return MTS_SUCCESS;
}

static mts_status_t bench_array_create(
    const void* array,
    const uintptr_t* shape,
    uintptr_t shape_count,
    mts_array_t fill_value,
    mts_array_t* new_array
) {
    const auto* bench = static_cast<const BenchArray*>(array);

    auto* created = new BenchArray();
    created->is_i32 = bench->is_i32;
    created->shape.assign(shape, shape + shape_count);

    if (created->is_i32) {
        created->i32_data = std::make_shared<std::vector<int32_t>>(product(created->shape), 0);
    }

    if (fill_value.destroy != nullptr) {
        fill_value.destroy(fill_value.ptr);
    }

    new_array->ptr = created;
    new_array->destroy = bench_array_destroy;
    new_array->origin = bench_array_origin;
    new_array->device = bench_array_device;
    new_array->dtype = bench_array_dtype;
    new_array->as_dlpack = bench_array_as_dlpack;
    new_array->shape = bench_array_shape;
    new_array->reshape = bench_array_reshape;
    new_array->swap_axes = bench_array_swap_axes;
    new_array->create = bench_array_create;
    new_array->copy = nullptr;
    new_array->move_data = nullptr;
    return MTS_SUCCESS;
}

static mts_status_t bench_array_copy(const void* array, DLDevice device, mts_array_t* new_array) {
    if (device.device_type != kDLCPU) {
        mts_set_last_error("can only copy benchmark array to CPU", "benchmark arrays", nullptr, nullptr);
        return MTS_CALLBACK_ERROR;
    }

    const auto* bench = static_cast<const BenchArray*>(array);
    auto* copied = new BenchArray(*bench);

    new_array->ptr = copied;
    new_array->destroy = bench_array_destroy;
    new_array->origin = bench_array_origin;
    new_array->device = bench_array_device;
    new_array->dtype = bench_array_dtype;
    new_array->as_dlpack = bench_array_as_dlpack;
    new_array->shape = bench_array_shape;
    new_array->reshape = bench_array_reshape;
    new_array->swap_axes = bench_array_swap_axes;
    new_array->create = bench_array_create;
    new_array->copy = bench_array_copy;
    new_array->move_data = nullptr;
    return MTS_SUCCESS;
}

static mts_status_t bench_array_move_data(void*, const void*, const mts_data_movement_t*, uintptr_t) {
    set_callback_error("move_data is not implemented for benchmark arrays");
    return MTS_CALLBACK_ERROR;
}

mts_array_t make_empty_array(std::vector<uintptr_t> shape) {
    auto* bench = new BenchArray();
    bench->is_i32 = false;
    bench->shape = std::move(shape);

    mts_array_t array = {};
    array.ptr = bench;
    array.destroy = bench_array_destroy;
    array.origin = bench_array_origin;
    array.device = bench_array_device;
    array.dtype = bench_array_dtype;
    array.as_dlpack = bench_array_as_dlpack;
    array.shape = bench_array_shape;
    array.reshape = bench_array_reshape;
    array.swap_axes = bench_array_swap_axes;
    array.create = bench_array_create;
    array.copy = bench_array_copy;
    array.move_data = bench_array_move_data;
    return array;
}

mts_array_t make_i32_array(
    std::shared_ptr<std::vector<int32_t>> data,
    std::vector<uintptr_t> shape
) {
    auto* bench = new BenchArray();
    bench->is_i32 = true;
    bench->shape = std::move(shape);
    bench->i32_data = std::move(data);

    mts_array_t array = {};
    array.ptr = bench;
    array.destroy = bench_array_destroy;
    array.origin = bench_array_origin;
    array.device = bench_array_device;
    array.dtype = bench_array_dtype;
    array.as_dlpack = bench_array_as_dlpack;
    array.shape = bench_array_shape;
    array.reshape = bench_array_reshape;
    array.swap_axes = bench_array_swap_axes;
    array.create = bench_array_create;
    array.copy = bench_array_copy;
    array.move_data = bench_array_move_data;
    return array;
}
