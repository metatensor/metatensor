#include <algorithm>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "./bench_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


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

std::string format_result_table(const std::vector<std::pair<std::string, BenchmarkResult>>& results) {
    auto out = std::ostringstream();
    out << '\n';

    constexpr int name_width = 45;
    out << std::left << std::setw(name_width) << "Function"
        << std::right << std::setw(12) << "Mean"
        << std::setw(12) << "Std"
        << std::setw(12) << "Min"
        << std::setw(12) << "Max"
        << '\n';
    out << std::string(name_width + 48, '-') << '\n';

    for (const auto& [name, result] : results) {
        auto [unit, scale] = guess_unit(result.mean);
        out << std::left << std::setw(name_width) << name
            << "     " << std::right << std::setw(6) << std::fixed << std::setprecision(2) << result.mean * scale << unit
            << "     " << std::right << std::setw(6) << std::fixed << std::setprecision(2) << result.std * scale << unit
            << "     " << std::right << std::setw(6) << std::fixed << std::setprecision(2) << result.min * scale << unit
            << "     " << std::right << std::setw(6) << std::fixed << std::setprecision(2) << result.max * scale << unit
            << '\n';
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

std::string format_result_json(const std::vector<std::pair<std::string, BenchmarkResult>>& results) {
    auto out = std::ostringstream();
    out << "{";
    bool first = true;
    for (const auto& [name, result] : results) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << escape_json(name) << "\":{"
            << "\"mean\":" << std::setprecision(17) << result.mean << ","
            << "\"std\":" << std::setprecision(17) << result.std << ","
            << "\"min\":" << std::setprecision(17) << result.min << ","
            << "\"max\":" << std::setprecision(17) << result.max
            << "}";
    }
    out << "}";
    return out.str();
}

CommandLineOptions CommandLineOptions::parse(int argc, const char* argv[]) {
    auto options = CommandLineOptions();
    for (int i = 1; i < argc; i++) {
        auto arg = std::string(argv[i]);
        if (arg == "--json") {
            options.json = true;
        } else if (arg == "--list") {
            options.list = true;
        } else if (arg == "--test") {
            options.test = true;
        } else {
            options.patterns.emplace_back(std::move(arg));
        }
    }

    if (options.patterns.empty()) {
        options.patterns.emplace_back(".*");
    }

    return options;
}

void run_benchmarks(
    const std::string& benchmark_name,
    const CommandLineOptions& options,
    const std::vector<std::pair<std::string, BenchFunction>>& benchmarks
) {
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

    std::optional<size_t> n_iters = options.test ? std::make_optional<size_t>(1) : std::nullopt;
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
            results.emplace_back(name, function(n_iters, n_warmup));
        } catch (const std::exception& error) {
            std::cerr << "error while running " << name << ": " << error.what() << "\n";
        }
    }

    if (options.json) {
        std::cout << format_result_json(results) << "\n";
    } else {
        std::cout << "Benchmark results for " << benchmark_name << ":\n";
        std::cout << format_result_table(results);
    }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


static const char* BENCH_ARRAY_ORIGIN = "metatensor.bench.classes-c";

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

static  void set_callback_error(const char* message) {
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
    return mts_register_data_origin(BENCH_ARRAY_ORIGIN, origin);
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

static mts_status_t bench_array_copy(const void* array, mts_array_t* new_array) {
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
