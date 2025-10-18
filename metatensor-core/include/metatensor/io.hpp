#pragma once

#include <cassert>
#include <cstring>
#include <string>
#include <vector>

#include <metatensor.h>

#include "./errors.hpp"
#include "./labels.hpp"
#include "./block.hpp"
#include "./tensor.hpp"


namespace metatensor {
namespace io {
    inline void save(const std::string& path, const TensorMap& tensor) {
        details::check_status(mts_tensormap_save(path.c_str(), tensor.as_mts_tensormap_t()));
    }

    template <typename Buffer>
    Buffer save_buffer(const TensorMap& tensor) {
        auto buffer = metatensor::io::save_buffer<std::vector<uint8_t>>(tensor);
        return Buffer(buffer.begin(), buffer.end());
    }

    template<>
    inline std::vector<uint8_t> save_buffer<std::vector<uint8_t>>(const TensorMap& tensor) {
        std::vector<uint8_t> buffer;

        auto* ptr = buffer.data();
        auto size = buffer.size();

        auto realloc = [](void* user_data, uint8_t*, uintptr_t new_size) {
            auto* buffer = reinterpret_cast<std::vector<uint8_t>*>(user_data);
            buffer->resize(new_size, '\0');
            return buffer->data();
        };

        details::check_status(mts_tensormap_save_buffer(
            &ptr,
            &size,
            &buffer,
            realloc,
            tensor.as_mts_tensormap_t()
        ));

        buffer.resize(size, '\0');

        return buffer;
    }

    /**************************************************************************/

    inline void save(const std::string& path, const TensorBlock& block) {
        details::check_status(mts_block_save(path.c_str(), block.as_mts_block_t()));
    }

    template <typename Buffer>
    Buffer save_buffer(const TensorBlock& block) {
        auto buffer = metatensor::io::save_buffer<std::vector<uint8_t>>(block);
        return Buffer(buffer.begin(), buffer.end());
    }

    template<>
    inline std::vector<uint8_t> save_buffer<std::vector<uint8_t>>(const TensorBlock& block) {
        std::vector<uint8_t> buffer;

        auto* ptr = buffer.data();
        auto size = buffer.size();

        auto realloc = [](void* user_data, uint8_t*, uintptr_t new_size) {
            auto* buffer = reinterpret_cast<std::vector<uint8_t>*>(user_data);
            buffer->resize(new_size, '\0');
            return buffer->data();
        };

        details::check_status(mts_block_save_buffer(
            &ptr,
            &size,
            &buffer,
            realloc,
            block.as_mts_block_t()
        ));

        buffer.resize(size, '\0');

        return buffer;
    }

    /**************************************************************************/

    inline void save(const std::string& path, const Labels& labels) {
        details::check_status(mts_labels_save(path.c_str(), labels.as_mts_labels_t()));
    }

    template <typename Buffer>
    Buffer save_buffer(const Labels& labels) {
        auto buffer = metatensor::io::save_buffer<std::vector<uint8_t>>(labels);
        return Buffer(buffer.begin(), buffer.end());
    }

    template<>
    inline std::vector<uint8_t> save_buffer<std::vector<uint8_t>>(const Labels& labels) {
        std::vector<uint8_t> buffer;

        auto* ptr = buffer.data();
        auto size = buffer.size();

        auto realloc = [](void* user_data, uint8_t*, uintptr_t new_size) {
            auto* buffer = reinterpret_cast<std::vector<uint8_t>*>(user_data);
            buffer->resize(new_size, '\0');
            return buffer->data();
        };

        details::check_status(mts_labels_save_buffer(
            &ptr,
            &size,
            &buffer,
            realloc,
            labels.as_mts_labels_t()
        ));

        buffer.resize(size, '\0');

        return buffer;
    }

    /**************************************************************************/
    /**************************************************************************/

    inline TensorMap load(
        const std::string& path,
        mts_create_array_callback_t create_array
    ) {
        auto* ptr = mts_tensormap_load(path.c_str(), create_array);
        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    inline TensorMap load_buffer(
        const uint8_t* buffer,
        size_t buffer_count,
        mts_create_array_callback_t create_array
    ) {
        auto* ptr = mts_tensormap_load_buffer(buffer, buffer_count, create_array);
        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    template <typename Buffer>
    TensorMap load_buffer(
        const Buffer& buffer,
        mts_create_array_callback_t create_array
    ) {
        static_assert(
            sizeof(typename Buffer::value_type) == sizeof(uint8_t),
            "`Buffer` must be a container of uint8_t or equivalent"
        );

        return metatensor::io::load_buffer(
            reinterpret_cast<const uint8_t*>(buffer.data()),
            buffer.size(),
            create_array
        );
    }

    /**************************************************************************/

    inline TensorBlock load_block(
        const std::string& path,
        mts_create_array_callback_t create_array
    ) {
        auto* ptr = mts_block_load(path.c_str(), create_array);
        details::check_pointer(ptr);
        return TensorBlock(ptr);
    }

    inline TensorBlock load_block_buffer(
        const uint8_t* buffer,
        size_t buffer_count,
        mts_create_array_callback_t create_array
    ) {
        auto* ptr = mts_block_load_buffer(buffer, buffer_count, create_array);
        details::check_pointer(ptr);
        return TensorBlock(ptr);
    }

    template <typename Buffer>
    TensorBlock load_block_buffer(
        const Buffer& buffer,
        mts_create_array_callback_t create_array
    ) {
        static_assert(
            sizeof(typename Buffer::value_type) == sizeof(uint8_t),
            "`Buffer` must be a container of uint8_t or equivalent"
        );

        return metatensor::io::load_block_buffer(
            reinterpret_cast<const uint8_t*>(buffer.data()),
            buffer.size(),
            create_array
        );
    }

    /**************************************************************************/

    inline Labels load_labels(const std::string& path) {
        mts_labels_t labels;
        std::memset(&labels, 0, sizeof(labels));

        details::check_status(mts_labels_load(
            path.c_str(), &labels
        ));

        return Labels(labels);
    }

    inline Labels load_labels_buffer(const uint8_t* buffer, size_t buffer_count) {
        mts_labels_t labels;
        std::memset(&labels, 0, sizeof(labels));

        details::check_status(mts_labels_load_buffer(
            buffer, buffer_count, &labels
        ));

        return Labels(labels);
    }

    template <typename Buffer>
    Labels load_labels_buffer(const Buffer& buffer) {
        static_assert(
            sizeof(typename Buffer::value_type) == sizeof(uint8_t),
            "`Buffer` must be a container of uint8_t or equivalent"
        );

        return metatensor::io::load_labels_buffer(
            reinterpret_cast<const uint8_t*>(buffer.data()),
            buffer.size()
        );
    }

} // namespace io
} // namespace metatensor
