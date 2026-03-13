#include <cassert>

#include <torch/version.h>
#include <torch/torch.h>

#include <metatensor.hpp>

#include "metatensor/torch/labels.hpp"
#include <ATen/DLConvertor.h>

#include "./utils.hpp"

using namespace metatensor_torch;

/// Data backing an mts_array_t that wraps a torch::Tensor for labels values.
///
/// This is distinct from TorchDataArray (which is for f64 block data) since
/// labels are i32 and only need origin/shape/copy/destroy.
struct TorchLabelsArrayData {
    torch::Tensor tensor;
    std::vector<uintptr_t> shape;
};

/// Origin ID for torch-backed labels arrays, distinct from TORCH_DATA_ORIGIN
/// (which is for f64 block data backed by TorchDataArray). Using a separate
/// origin prevents type confusion if an mts_array_t with TORCH_DATA_ORIGIN
/// were ever passed as a labels values array.
static mts_data_origin_t TORCH_LABELS_ORIGIN = 0;

static mts_status_t torch_labels_origin(const void*, mts_data_origin_t* origin) {
    static auto REGISTRATION = []{
        auto name = std::string("metatensor_torch::TorchLabelsArrayData");
        mts_register_data_origin(name.c_str(), &TORCH_LABELS_ORIGIN);
        return 0;
    }();
    (void)REGISTRATION;
    *origin = TORCH_LABELS_ORIGIN;
    return MTS_SUCCESS;
}

static mts_status_t torch_labels_device(const void* ptr, DLDevice* device) {
    auto* data = static_cast<const TorchLabelsArrayData*>(ptr);
    auto torch_dev = data->tensor.device();

    DLDeviceType dl_type;
    switch (torch_dev.type()) {
    case torch::DeviceType::CPU:    dl_type = kDLCPU;    break;
    case torch::DeviceType::CUDA:   dl_type = kDLCUDA;   break;
    case torch::DeviceType::HIP:    dl_type = kDLROCM;   break;
    case torch::DeviceType::MPS:    dl_type = kDLMetal;   break;
    case torch::DeviceType::XPU:    dl_type = kDLOneAPI;  break;
    case torch::DeviceType::XLA:    dl_type = kDLTrn;     break;
    case torch::DeviceType::Vulkan: dl_type = kDLVulkan;  break;
    case torch::DeviceType::Meta:   dl_type = kDLExtDev;  break;
    default:
        return 1;
    }

    device->device_type = dl_type;
    device->device_id = static_cast<int32_t>(torch_dev.index() < 0 ? 0 : torch_dev.index());
    return MTS_SUCCESS;
}

static void torch_labels_dlpack_deleter(DLManagedTensorVersioned* self) {
    if (self != nullptr) {
        auto* legacy_tensor = static_cast<DLManagedTensor*>(self->manager_ctx);
        if (legacy_tensor != nullptr && legacy_tensor->deleter != nullptr) {
            legacy_tensor->deleter(legacy_tensor);
        }
        delete self;
    }
}

static mts_status_t torch_labels_as_dlpack(
    void* ptr,
    DLManagedTensorVersioned** dl_managed_tensor,
    DLDevice device,
    const int64_t* stream,
    DLPackVersion max_version
) {
    try {
        auto* data = static_cast<TorchLabelsArrayData*>(ptr);

        DLPackVersion mta_version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        if (max_version.major != mta_version.major || max_version.minor < mta_version.minor) {
            return 1;
        }

        // Convert DLDevice to torch::Device
        torch::DeviceType target_type;
        switch (device.device_type) {
        case kDLCPU:         target_type = torch::DeviceType::CPU;    break;
        case kDLCUDA:        target_type = torch::DeviceType::CUDA;   break;
        case kDLCUDAHost:    target_type = torch::DeviceType::CPU;    break;
        case kDLCUDAManaged: target_type = torch::DeviceType::CUDA;   break;
        case kDLROCM:        target_type = torch::DeviceType::HIP;    break;
        case kDLROCMHost:    target_type = torch::DeviceType::CPU;    break;
        case kDLMetal:       target_type = torch::DeviceType::MPS;    break;
        case kDLOneAPI:      target_type = torch::DeviceType::XPU;    break;
        case kDLTrn:         target_type = torch::DeviceType::XLA;    break;
        case kDLVulkan:      target_type = torch::DeviceType::Vulkan; break;
        default:
            return 1;
        }
        torch::Device target_device(target_type);

        torch::Tensor tensor_to_pack = data->tensor;
        if (tensor_to_pack.device() != target_device) {
            tensor_to_pack = tensor_to_pack.to(target_device, /*non_blocking=*/false);
        }

        if (tensor_to_pack.is_cuda()) {
            if (stream != nullptr && *stream != 0) {
                return 1;
            }
        } else if (tensor_to_pack.is_cpu() && stream != nullptr) {
            return 1;
        }

        DLManagedTensor* legacy_tensor = at::toDLPack(tensor_to_pack);

        if (legacy_tensor->dl_tensor.device.device_type != device.device_type ||
            legacy_tensor->dl_tensor.device.device_id != device.device_id) {
            if (legacy_tensor->deleter != nullptr) {
                legacy_tensor->deleter(legacy_tensor);
            }
            return 1;
        }

        auto* versioned = new DLManagedTensorVersioned();
        versioned->version = mta_version;
        versioned->manager_ctx = legacy_tensor;
        versioned->deleter = torch_labels_dlpack_deleter;
        versioned->flags = 0;
        versioned->dl_tensor = legacy_tensor->dl_tensor;

        *dl_managed_tensor = versioned;
        return MTS_SUCCESS;
    } catch (...) {
        // Catch exceptions (e.g. Meta tensor -> CPU transfer) to prevent
        // them from propagating through the C callback boundary, which
        // would cause std::terminate.
        return 1;
    }
}

static mts_status_t torch_labels_shape(const void* ptr, const uintptr_t** shape, uintptr_t* shape_count) {
    auto* data = static_cast<const TorchLabelsArrayData*>(ptr);
    *shape = data->shape.data();
    *shape_count = data->shape.size();
    return MTS_SUCCESS;
}

static mts_status_t torch_labels_copy(const void* ptr, mts_array_t* new_array) {
    auto* data = static_cast<const TorchLabelsArrayData*>(ptr);
    auto* cloned = new TorchLabelsArrayData{data->tensor.clone(), data->shape};
    std::memset(new_array, 0, sizeof(mts_array_t));
    new_array->ptr = cloned;
    new_array->origin = torch_labels_origin;
    new_array->device = torch_labels_device;
    new_array->as_dlpack = torch_labels_as_dlpack;
    new_array->shape = torch_labels_shape;
    new_array->copy = torch_labels_copy;
    new_array->destroy = [](void* p) { delete static_cast<TorchLabelsArrayData*>(p); };
    return MTS_SUCCESS;
}

/// Create an mts_array_t wrapping a torch::Tensor for use as labels values.
static mts_array_t torch_tensor_to_labels_mts_array(torch::Tensor tensor) {
    auto shape = std::vector<uintptr_t>();
    for (auto s: tensor.sizes()) {
        shape.push_back(static_cast<uintptr_t>(s));
    }
    auto* data = new TorchLabelsArrayData{std::move(tensor), std::move(shape)};

    mts_array_t array;
    std::memset(&array, 0, sizeof(array));
    array.ptr = data;
    array.origin = torch_labels_origin;
    array.device = torch_labels_device;
    array.as_dlpack = torch_labels_as_dlpack;
    array.shape = torch_labels_shape;
    array.copy = torch_labels_copy;
    array.destroy = [](void* p) { delete static_cast<TorchLabelsArrayData*>(p); };
    return array;
}

/// Check that `values` is a `shape_length`-dimensional array of 32-bit
/// integers, or convert it to be so.
static torch::Tensor normalize_int32_tensor(torch::Tensor values, size_t shape_length, const std::string& context) {
    // if the tensor is empty, make sure the shape is (0, len(sample_names))
    if (values.numel() == 0) {
        values = torch::empty(values.sizes(), torch::TensorOptions().dtype(torch::kInt32).device(values.device()));
    }

    if (!torch::can_cast(values.scalar_type(), torch::kI32)) {
        C10_THROW_ERROR(ValueError,
            context + " must be a Tensor of 32-bit integers"
        );
    }

    if (values.sizes().size() != shape_length) {
        C10_THROW_ERROR(ValueError,
            context + " must be a " + std::to_string(shape_length) + "D Tensor"
        );
    }

    return values.to(torch::kI32);
}

static torch::Tensor initializer_list_to_tensor(
    const std::vector<std::initializer_list<int32_t>>& values,
    size_t size
) {
    auto vector = std::vector<int32_t>();

    auto count = values.size();
    vector.reserve(count * size);
    for (auto row: values) {
        if (row.size() != size) {
            C10_THROW_ERROR(ValueError,
                "invalid size for row: expected " + std::to_string(size) +
                " got " + std::to_string(row.size())
            );
        }

        for (auto entry: row) {
            vector.emplace_back(entry);
        }
    }

    auto tensor = torch::tensor(vector);
    return tensor.reshape({static_cast<int64_t>(count), static_cast<int64_t>(size)});
}

std::vector<std::string> metatensor_torch::details::normalize_names(torch::IValue names, const std::string& argument_name) {
    auto results = std::vector<std::string>();
    if (names.isString()) {
        results.push_back(names.toStringRef());
    } else if (names.isList()) {
        const auto& names_list = names.toListRef();
        for (const auto& name: names_list) {
            if (!name.isString()) {
                C10_THROW_ERROR(TypeError,
                    argument_name + " must be a list of strings, got element with type '"
                    + name.type()->str() + "' instead"
                );
            }
            results.push_back(name.toStringRef());
        }
    } else if (names.isTuple()) {
        for (const auto& name: names.toTupleRef().elements()) {
            if (!name.isString()) {
                C10_THROW_ERROR(TypeError,
                    argument_name + " must be a tuple of strings, got element with type '"
                    + name.type()->str() + "' instead"
                );
            }
            results.push_back(name.toStringRef());
        }
    } else {
        C10_THROW_ERROR(TypeError,
            argument_name + " must be a string, list of strings or tuple of string, got '"
            + names.type()->str() + "' instead"
        );
    }
    return results;
}

LabelsHolder::LabelsHolder(
    std::vector<std::string> names,
    torch::Tensor values,
    metatensor::Labels labels
):
    names_(std::move(names)),
    values_(std::move(values)),
    labels_(std::move(labels))
{
    // basic checks in debug mode to make sure everything is fine
    assert(values_.sizes().size() == 2);
    assert(values_.size(0) == labels_->count());
    assert(values_.size(1) == labels_->size());
    assert(names_.size() == labels_->size());
    for (size_t i=0; i<names.size(); i++) {
        assert(names_[i] == labels_->names()[i]);
    }
    assert(values_.scalar_type() == torch::kInt32);
}

LabelsHolder::LabelsHolder(torch::IValue names, torch::Tensor values):
    names_(details::normalize_names(names, "names")),
    values_(normalize_int32_tensor(values, 2, "Labels values")),
    labels_(torch::nullopt)
{
    if (values_.sizes()[1] != names_.size()) {
        C10_THROW_ERROR(ValueError,
            "invalid Labels: the names must have an entry for each column of the array"
        );
    }

    auto array = torch_tensor_to_labels_mts_array(values_);
    if (values_.is_cpu()) {
        labels_ = metatensor::Labels(names_, std::move(array), metatensor::from_array{});
    } else {
        labels_ = metatensor::Labels(names_, std::move(array), metatensor::assume_unique{});
    }
}

LabelsHolder::LabelsHolder(torch::IValue names, torch::Tensor values, metatensor::assume_unique):
    names_(details::normalize_names(names, "names")),
    values_(normalize_int32_tensor(values, 2, "Labels values")),
    labels_(torch::nullopt)
{
    if (values_.sizes()[1] != names_.size()) {
        C10_THROW_ERROR(ValueError,
            "invalid Labels: the names must have an entry for each column of the array"
        );
    }

    auto array = torch_tensor_to_labels_mts_array(values_);
    labels_ = metatensor::Labels(names_, std::move(array), metatensor::assume_unique{});
}

Labels LabelsHolder::create(
    std::vector<std::string> names,
    const std::vector<std::initializer_list<int32_t>>& values
) {
    auto torch_values = initializer_list_to_tensor(values, names.size());
    return torch::make_intrusive<LabelsHolder>(std::move(names), std::move(torch_values));
}

LabelsHolder::LabelsHolder(metatensor::Labels labels): labels_(std::move(labels)) {
    // extract the names
    for (const auto* name: this->labels_->names()) {
        this->names_.emplace_back(name);
    }

    // check if the values array is backed by a torch tensor
    auto array = this->labels_->values_array();
    mts_data_origin_t origin = 0;
    if (array.origin != nullptr) {
        array.origin(array.ptr, &origin);
    }

    if (origin == TORCH_LABELS_ORIGIN && array.ptr != nullptr) {
        // recover the torch tensor from the values array
        auto* data = static_cast<TorchLabelsArrayData*>(array.ptr);
        this->values_ = data->tensor;
    } else {
        auto clone = this->labels_.value();
        // otherwise create a new tensor which shares memory with the Labels.
        const auto* values = clone.values().data();
        auto sizes = std::vector<int64_t>{
            static_cast<int64_t>(clone.count()),
            static_cast<int64_t>(clone.size()),
        };

        this->values_ = torch::from_blob(
            // This should really be a `const int32_t*`, but we can not prevent
            // writing to this tensor since torch does not support read-only tensor:
            // https://github.com/pytorch/pytorch/issues/44027
            const_cast<int32_t*>(values),
            sizes,
            // capture `clone` inside the torch::Tensor custom deleter to
            // keep the corresponding data alive as long as the tensor might be
            [clone=std::move(clone)](void*) mutable {
                // when running this function (i.e. when destroying the
                // torch::Tensor), we move the cloned Labels & let them go out
                // of scope, releasing the corresponding memory.
                auto _ = std::move(clone);
            },
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
        );
    }
}


Labels LabelsHolder::single() {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto values = torch::tensor({0}, options).reshape({1, 1});
    return torch::make_intrusive<LabelsHolder>("_", std::move(values));
}


Labels LabelsHolder::empty(torch::IValue names_ivalue) {
    auto names = details::normalize_names(names_ivalue, "empty");
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto values = torch::tensor(std::vector<int>(), options).reshape({0, static_cast<int64_t>(names.size())});
    return torch::make_intrusive<LabelsHolder>(std::move(names), std::move(values));
}


Labels LabelsHolder::range(std::string name, int64_t end) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto values = torch::arange(end, options).reshape({end, 1});
    return torch::make_intrusive<LabelsHolder>(name, std::move(values));
}

torch::Tensor LabelsHolder::column(std::string dimension) {
    auto it = std::find(std::begin(names_), std::end(names_), dimension);

    if (it == std::end(names_)) {
        C10_THROW_ERROR(ValueError,
            "'" + dimension + "' not found in the dimensions of these Labels"
        );
    }

    int64_t index = std::distance(std::begin(names_), it);
    return values_.index({torch::indexing::Slice(), index});
}

const metatensor::Labels& LabelsHolder::as_metatensor() const {
    return labels_.value();
}

Labels LabelsHolder::append(std::string name, torch::Tensor values) const {
    return this->insert(this->size(), std::move(name), std::move(values));
}


Labels LabelsHolder::insert(int64_t index, std::string name, torch::Tensor values) const {
    if (index < 0 || index > this->size()) {
        C10_THROW_ERROR(
            IndexError,
            "index " + std::to_string(index) + " is out of bounds"
        );
    }

    auto new_names = this->names();

    auto it = std::begin(new_names) + index;
    new_names.insert(it, std::move(name));

    if (values.sizes().size() != 1) {
        C10_THROW_ERROR(
            ValueError,
            "`values` must be a 1D tensor"
        );
    }

    if (values.size(0) != this->count()) {
        C10_THROW_ERROR(
            ValueError,
            "the new `values` contains " + std::to_string(values.size(0)) + " entries, "
            "but the Labels contains " + std::to_string(this->count())
        );
    }

    auto old_values = this->values();
    auto first = old_values.index({torch::indexing::Slice(), torch::indexing::Slice(0, index)});
    auto second = old_values.index({torch::indexing::Slice(), torch::indexing::Slice(index)});

    auto new_values = torch::hstack({first, values.reshape({values.size(0), 1}), second});

    return torch::make_intrusive<LabelsHolder>(std::move(new_names), std::move(new_values));
}


Labels LabelsHolder::permute(std::vector<int64_t> dimensions_indexes) const {
    auto names = this->names();

    if (dimensions_indexes.size() != names.size()) {
        C10_THROW_ERROR(
            ValueError,
            "the length of `dimensions_indexes` (" + std::to_string(dimensions_indexes.size()) + ") "
            "does not match the number of dimensions in the Labels (" + std::to_string(names.size()) + ")"
        );
    }

    auto new_names = std::vector<std::string>();

    for (auto index : dimensions_indexes) {
        if (index < 0) {
            index += static_cast<int64_t>(names.size());
        }
        if (index >= names.size()) {
            C10_THROW_ERROR(
                IndexError,
                "out of range index "  + std::to_string(index) + " for labels dimensions (" +
                std::to_string(names.size()) + ")"
            );
        }
        new_names.push_back(names[index]);
    }

    auto new_values = this->values().index({torch::indexing::Slice(), torch::tensor(dimensions_indexes)});

    return torch::make_intrusive<LabelsHolder>(std::move(new_names), std::move(new_values));
}


Labels LabelsHolder::remove(std::string name) const {
    auto new_names = this->names();

    auto it = std::find(std::begin(new_names), std::end(new_names), name);
    if (it == std::end(new_names)) {
        C10_THROW_ERROR(
            ValueError,
            "'" + name + "' not found in the dimensions of these Labels"
        );
    }

    auto column_index = it - std::begin(new_names);
    new_names.erase(it);

    auto values = this->values();
    auto first = values.index({torch::indexing::Slice(), torch::indexing::Slice(0, column_index)});
    auto second = values.index({torch::indexing::Slice(), torch::indexing::Slice(column_index + 1)});
    auto new_values = torch::hstack({first, second});

    return torch::make_intrusive<LabelsHolder>(std::move(new_names), std::move(new_values));
}


Labels LabelsHolder::rename(std::string old_name, std::string new_name) const {
    auto new_names = this->names();

    auto it = std::find(std::begin(new_names), std::end(new_names), old_name);
    if (it == std::end(new_names)) {
        C10_THROW_ERROR(
            ValueError,
            "'" + old_name + "' not found in the dimensions of these Labels"
        );
    }
    auto column_index = it - std::begin(new_names);

    new_names[column_index] = std::move(new_name);
    return torch::make_intrusive<LabelsHolder>(std::move(new_names), this->values());
}

Labels LabelsHolder::to(torch::IValue device_ivalue, bool non_blocking) const {
    auto device = this->device();
    if (device_ivalue.isNone()) {
        // nothing to do
    } else if (device_ivalue.isString()) {
        device = torch::Device(device_ivalue.toStringRef());
    } else if (device_ivalue.isDevice()) {
        device = device_ivalue.toDevice();
    } else {
        C10_THROW_ERROR(TypeError,
            "'device' must be a string or a torch.device, got '" + device_ivalue.type()->str() + "' instead"
        );
    }
    return this->to(device, non_blocking);
}

Labels LabelsHolder::to(torch::Device device, bool non_blocking) const {
    if (device == values_.device()) {
        // return the same object
        return torch::make_intrusive<LabelsHolder>(*this);
    } else if (device.is_meta()) {
        auto new_values = values_.to(device, non_blocking);
        // Create Rust Labels with Meta-backed array (preserves device in round-trips)
        auto meta_array = torch_tensor_to_labels_mts_array(new_values);
        auto new_labels = metatensor::Labels(
            this->names(), std::move(meta_array), metatensor::assume_unique{}
        );
        // Pre-fill CPU values from the original Labels so Rust never needs to
        // materialize from the Meta array (which has no storage)
        auto orig = this->as_metatensor();
        auto raw = orig.as_mts_labels_t();
        const int32_t* orig_values_ptr = nullptr;
        size_t orig_count = 0;
        metatensor::details::check_status(
            mts_labels_values(raw, &orig_values_ptr, &orig_count)
        );
        new_labels.set_cached_values(orig_values_ptr, orig_count);

        return torch::make_intrusive<LabelsHolder>(
            this->names(), std::move(new_values), std::move(new_labels)
        );
    } else {
        auto new_values = values_.to(device, non_blocking);

        // create new Labels from the moved tensor's array
        auto array = torch_tensor_to_labels_mts_array(new_values);
        auto new_labels = metatensor::Labels(
            this->names(), std::move(array), metatensor::assume_unique{}
        );

        return torch::make_intrusive<LabelsHolder>(
            this->names(),
            std::move(new_values),
            std::move(new_labels)
        );
    }
}

torch::optional<int64_t> LabelsHolder::position(torch::IValue entry) const {
    const auto& labels = this->as_metatensor();

    int64_t position = -1;
    if (entry.isCustomClass()) {
        const auto& labels_entry = entry.toCustomClass<LabelsEntryHolder>();
        auto values = labels_entry->values().to(torch::kCPU).contiguous();
        position = labels.position(
            static_cast<const int32_t*>(values.data_ptr()),
            values.size(0)
        );
    } else if (entry.isTensor()) {
        auto tensor = normalize_int32_tensor(entry.toTensor(), 1, "entry passed to Labels::position");
        tensor = tensor.to(torch::kCPU).contiguous();
        position = labels.position(
            static_cast<const int32_t*>(tensor.data_ptr()),
            tensor.size(0)
        );
    } else if (entry.isIntList()) {
        auto int32_values = std::vector<int32_t>();
        for (const auto& value: entry.toIntList()) {
            int32_values.push_back(static_cast<int32_t>(value));
        }
        position = labels.position(int32_values);
    } else if (entry.isList()) {
        auto int32_values = std::vector<int32_t>();
        for (const auto& value: entry.toListRef()) {
            if (value.isInt()) {
                int32_values.push_back(static_cast<int32_t>(value.toInt()));
            } else {
                C10_THROW_ERROR(TypeError,
                    "list parameter to Labels::positions must be list of integers, "
                    "got element with type '" + entry.type()->str() + "'"
                );
            }
        }
        position = labels.position(int32_values);
    } else if (entry.isTuple()) {
        auto int32_values = std::vector<int32_t>();
        for (const auto& value: entry.toTupleRef().elements()) {
            if (value.isInt()) {
                int32_values.push_back(static_cast<int32_t>(value.toInt()));
            } else {
                C10_THROW_ERROR(TypeError,
                    "tuple parameter to Labels::positions must be a tuple of integers, "
                    "got element with type '" + entry.type()->str() + "'"
                );
            }
        }
        position = labels.position(int32_values);
    } else {
        C10_THROW_ERROR(TypeError,
            "parameter to Labels::positions must be a LabelsEntry, tensor, or list/tuple of integers, "
            "got '" + entry.type()->str() + "' instead"
        );
    }

    if (position == -1) {
        return {};
    } else {
       return position;
    }
}

Labels LabelsHolder::set_union(const Labels& other) const {
    auto device = this->values_.device();
    if (device != other->values_.device()) {
        C10_THROW_ERROR(ValueError,
            "device mismatch in `Labels.union`: got '" + device.str() +
            "' and '" + other->values_.device().str() + "'"
        );
    }

    auto result = LabelsHolder(labels_->set_union(other->labels_.value()));
    return result.to(device);
}

std::tuple<Labels, torch::Tensor, torch::Tensor> LabelsHolder::union_and_mapping(const Labels& other) const {
    auto device = this->values_.device();
    if (device != other->values_.device()) {
        C10_THROW_ERROR(ValueError,
            "device mismatch in `Labels.union_and_mapping`: got '" + device.str() +
            "' and '" + other->values_.device().str() + "'"
        );
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto first_mapping = torch::zeros({this->count()}, options);
    auto second_mapping = torch::zeros({other->count()}, options);

    auto result = LabelsHolder(labels_->set_union(
        other->labels_.value(),
        first_mapping.data_ptr<int64_t>(),
        first_mapping.size(0),
        second_mapping.data_ptr<int64_t>(),
        second_mapping.size(0)
    ));

    return std::make_tuple<Labels, torch::Tensor, torch::Tensor>(
        result.to(device),
        first_mapping.to(device),
        second_mapping.to(device)
    );
}

Labels LabelsHolder::set_intersection(const Labels& other) const {
    auto device = this->values_.device();
    if (device != other->values_.device()) {
        C10_THROW_ERROR(ValueError,
            "device mismatch in `Labels.intersection`: got '" + device.str() +
            "' and '" + other->values_.device().str() + "'"
        );
    }

    auto result = LabelsHolder(labels_->set_intersection(other->labels_.value()));
    return result.to(device);
}

std::tuple<Labels, torch::Tensor, torch::Tensor> LabelsHolder::intersection_and_mapping(const Labels& other) const {
    auto device = this->values_.device();
    if (device != other->values_.device()) {
        C10_THROW_ERROR(ValueError,
            "device mismatch in `Labels.intersection_and_mapping`: got '" + device.str() +
            "' and '" + other->values_.device().str() + "'"
        );
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto first_mapping = torch::zeros({this->count()}, options);
    auto second_mapping = torch::zeros({other->count()}, options);

    auto result = LabelsHolder(labels_->set_intersection(
        other->labels_.value(),
        first_mapping.data_ptr<int64_t>(),
        first_mapping.size(0),
        second_mapping.data_ptr<int64_t>(),
        second_mapping.size(0)
    ));

    return std::make_tuple<Labels, torch::Tensor, torch::Tensor>(
        result.to(device),
        first_mapping.to(device),
        second_mapping.to(device)
    );
}

Labels LabelsHolder::set_difference(const Labels& other) const {
    auto device = this->values_.device();
    if (device != other->values_.device()) {
        C10_THROW_ERROR(ValueError,
            "device mismatch in `Labels.difference`: got '" + device.str() +
            "' and '" + other->values_.device().str() + "'"
        );
    }

    auto result = LabelsHolder(labels_->set_difference(other->labels_.value()));
    return result.to(device);
}

std::tuple<Labels, torch::Tensor> LabelsHolder::difference_and_mapping(const Labels& other) const {
    auto device = this->values_.device();
    if (device != other->values_.device()) {
        C10_THROW_ERROR(ValueError,
            "device mismatch in `Labels.difference_and_mapping`: got '" + device.str() +
            "' and '" + other->values_.device().str() + "'"
        );
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto mapping = torch::zeros({this->count()}, options);

    auto result = LabelsHolder(labels_->set_difference(
        other->labels_.value(),
        mapping.data_ptr<int64_t>(),
        mapping.size(0)
    ));

    return std::make_tuple<Labels, torch::Tensor>(
        result.to(device),
        mapping.to(device)
    );
}

torch::Tensor LabelsHolder::select(const Labels& selection) const {
    auto device = this->values_.device();
    if (device != selection->values_.device()) {
        C10_THROW_ERROR(ValueError,
            "device mismatch in `Labels.select`: got '" + device.str() +
            "' and '" + selection->values_.device().str() + "'"
        );
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto selected = torch::zeros({this->count()}, options);
    auto selected_count = static_cast<size_t>(selected.size(0));

    if (this->count() == 0) {
        return selected;
    }

    labels_->select(
        selection->labels_.value(),
        selected.data_ptr<int64_t>(),
        &selected_count
    );

    selected.resize_({static_cast<int64_t>(selected_count)});

    return selected;
}

struct LabelsPrintData {
    LabelsPrintData(const std::vector<std::string>& names) {
        for (const auto& name: names) {
            // use at least one space on each side of the name
            this->widths.push_back(name.size() + 2);
        }
    }

    /// widths of each column
    std::vector<size_t> widths;
    /// first half of the values
    std::vector<std::vector<std::string>> values_first;
    /// second half of the values
    std::vector<std::vector<std::string>> values_second;

    void add_values_first(torch::Tensor entry) {
        assert(entry.sizes().size() == 1);
        assert(this->widths.size() == entry.size(0));

        auto n_elements = this->widths.size();
        auto strings = std::vector<std::string>();
        strings.reserve(n_elements);

        for (int i=0; i<n_elements; i++) {
            auto entry_str = std::to_string(entry[i].item<int32_t>());
            this->widths[i] = std::max(entry_str.size() + 2, this->widths[i]);

            strings.emplace_back(std::move(entry_str));
        }

        this->values_first.emplace_back(std::move(strings));
    }

    void add_values_second(torch::Tensor entry) {
        assert(entry.sizes().size() == 1);
        assert(this->widths.size() == entry.size(0));

        auto n_elements = this->widths.size();
        auto strings = std::vector<std::string>();
        strings.reserve(n_elements);

        for (int i=0; i<n_elements; i++) {
            auto entry_str = std::to_string(entry[i].item<int32_t>());
            this->widths[i] = std::max(entry_str.size() + 2, this->widths[i]);

            strings.emplace_back(std::move(entry_str));
        }

        this->values_second.emplace_back(std::move(strings));
    }
};

void print_string_center(std::ostringstream& output, std::string string, size_t width, bool last) {
    assert(string.length() < width);

    auto delta = width - string.size();
    auto n_before = delta / 2;
    auto n_after = delta - n_before;
    string.insert(0, n_before, ' ');

    if (!last) {
        // don't add spaces after the last element
        string.insert(string.size(), n_after, ' ');
    }

    output << string;
}

std::string LabelsHolder::print(int64_t max_entries, int64_t indent) const {
    auto print_data = LabelsPrintData(names_);

    auto n_elements = this->count();
    if (max_entries < 0 || n_elements <= max_entries) {
        for (int i=0; i<n_elements; i++) {
            print_data.add_values_first(values_[i]);
        }
    } else {
        if (max_entries < 2) {
            max_entries = 2;
        }

        auto n_after = max_entries / 2;
        auto n_before = max_entries - n_after;

        for (int i=0; i<n_before; i++) {
            print_data.add_values_first(values_[i]);
        }

        for (int64_t i=(n_elements - n_after); i<n_elements; i++) {
            print_data.add_values_second(values_[i]);
        }
    }

    auto output = std::ostringstream();
    auto indent_str = std::string(indent, ' ');

    auto n_dimensions = this->size();
    for (int i=0; i<n_dimensions; i++) {
        auto last = i == n_dimensions - 1;
        print_string_center(output, names_[i], print_data.widths[i], last);
    }
    output << '\n';

    for (auto strings: std::move(print_data.values_first)) {
        output << indent_str;
        for (int i=0; i<n_dimensions; i++) {
            auto last = i == n_dimensions - 1;
            print_string_center(output, std::move(strings[i]), print_data.widths[i], last);
        }
        output << '\n';
    }


    if (!print_data.values_second.empty()) {
        size_t half_header_widths = 0;
        for (auto w: print_data.widths) {
            half_header_widths += w;
        }
        half_header_widths /= 2;

        if (half_header_widths > 3) {
            // 3 characters in '...'
            half_header_widths -= 3;
        }
        output << indent_str << std::string(half_header_widths + 1, ' ') << "...\n";

        for (auto strings: std::move(print_data.values_second)) {
            output << indent_str;
            for (int i=0; i<n_dimensions; i++) {
                auto last = i == n_dimensions - 1;
                print_string_center(output, std::move(strings[i]), print_data.widths[i], last);
            }
            output << '\n';
        }
    }

    // remove the final \n
    auto output_string = output.str();
    assert(output_string[output_string.size() - 1] == '\n');
    output_string.pop_back();

    return output_string;
}

std::string LabelsHolder::str() const {
    auto output = std::ostringstream();
    output << "Labels(\n   " << this->print(4, 3) << "\n)";
    return output.str();
}

std::string LabelsHolder::repr() const {
    auto output = std::ostringstream();
    output << "Labels(\n   " << this->print(-1, 3) << "\n)";
    return output.str();
}


Labels LabelsHolder::load(const std::string& path) {
    return torch::make_intrusive<LabelsHolder>(
        LabelsHolder(metatensor::io::load_labels(path))
    );
}


Labels LabelsHolder::load_buffer(torch::Tensor buffer) {
    if (buffer.scalar_type() != torch::kUInt8) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a tensor of uint8, not " +
            scalar_type_name(buffer.scalar_type())
        );
    }

    if (buffer.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a 1-dimensional tensor"
        );
    }

    return torch::make_intrusive<LabelsHolder>(
        LabelsHolder(metatensor::io::load_labels_buffer(
            buffer.data_ptr<uint8_t>(),
            static_cast<size_t>(buffer.size(0))
        ))
    );
}


void LabelsHolder::save(const std::string& path) const {
    metatensor::io::save(path, this->as_metatensor());
}

torch::Tensor LabelsHolder::save_buffer() const {
    auto buffer = metatensor::io::save_buffer(this->as_metatensor());
    // move the buffer to the heap so it can escape this function
    // `torch::from_blob` does not take ownership of the data,
    // so we need to register a custom deleter to clean up when
    // the tensor is no longer used
    auto* buffer_data = new std::vector<uint8_t>(std::move(buffer));

    auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCPU);
    auto deleter = [=](void* data) {
        delete buffer_data;
    };

    // use a tensor of bytes to store the data
    return torch::from_blob(
        buffer_data->data(),
        {static_cast<int64_t>(buffer_data->size())},
        deleter,
        options
    );
}

/******************************************************************************/

LabelsEntryHolder::LabelsEntryHolder(Labels labels, int64_t index):
    labels_(std::move(labels))
{
    auto size = labels_->values().size(0);
    if (index < -size || index >= size) {
        // We prefer not using `C10_THROW_ERROR` here since this is an expected
        // error (it will and must happen once per iteration over the labels),
        // and `C10_THROW_ERROR` is quite expensive since it construct a full
        // backtrace of the C++ code.
        //
        // Unfortunately, the `IndexError` constructor is not exported on
        // Windows, so we can not call it. In this case, we let the indexing
        // code below construct and throw the full error.
#ifndef _WIN32
        std::ostringstream ss;
        ss << "out of range for tensor of size " << labels_->values().sizes();
        ss << " at dimension 0";

#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 4
        throw torch::IndexError(ss.str(), /*backtrace=*/nullptr);
#else
        throw torch::IndexError(ss.str(), /*backtrace=*/"<no backtrace>");
#endif

#endif
    }

    values_ = labels_->values()[index];
}

std::string LabelsEntryHolder::print() const {
    auto output = std::stringstream();

    output << "(";
    for (int64_t i=0; i<this->size(); i++) {
        output << this->names()[i] << "=" << values_[i].item<int32_t>();

        if (i < this->size() - 1) {
            output << ", ";
        }
    }
    output << ")";

    return output.str();
}

std::string LabelsEntryHolder::repr() const {
    return "LabelsEntry" + this->print();
}

int32_t LabelsEntryHolder::operator[](const std::string& name) const {
    const auto& names = labels_->names();
    auto it = std::find(std::begin(names), std::end(names), name);
    if (it == std::end(names)) {
        C10_THROW_ERROR(ValueError,
            "'" + name + "' not found in the dimensions of this LabelsEntry"
        );
    }

    auto index = std::distance(std::begin(names), it);
    return this->operator[](index);
}


int64_t LabelsEntryHolder::getitem(torch::IValue index) const {
    if (index.isInt()) {
        return static_cast<int64_t>(this->operator[](index.toInt()));
    } else if (index.isString()) {
        return static_cast<int64_t>(this->operator[](index.toStringRef()));
    } else {
        C10_THROW_ERROR(TypeError,
            "LabelsEntry can only be indexed by int or str, got '"
            + index.type()->str() + "' instead"
        );
    }
}
