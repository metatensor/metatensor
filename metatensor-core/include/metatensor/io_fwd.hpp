#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <metatensor.h>

#include "./arrays.hpp"

namespace metatensor {
    class TensorMap;
    class TensorBlock;
    class Labels;

    namespace io {
        /// Save a `TensorMap` to the file at `path`.
        ///
        /// If the file exists, it will be overwritten. The recomended file
        /// extension when saving data is `.mts`, to prevent confusion with generic
        /// `.npz` files.
        ///
        /// `TensorMap` are serialized using numpy's NPZ format, i.e. a ZIP file
        /// without compression (storage method is `STORED`), where each file is
        /// stored as a `.npy` array. See the C API documentation for more
        /// information on the format.
        void save(const std::string& path, const TensorMap& tensor);

        /// Save a `TensorMap` to an in-memory buffer.
        ///
        /// The `Buffer` template parameter can be set to any type that can be
        /// constructed from a pair of iterator over `std::vector<uint8_t>`.
        template <typename Buffer = std::vector<uint8_t>>
        Buffer save_buffer(const TensorMap& tensor);

        template<>
        std::vector<uint8_t> save_buffer<std::vector<uint8_t>>(const TensorMap& tensor);

        /**************************************************************************/

        /// Save a `TensorBlock` to the file at `path`.
        ///
        /// If the file exists, it will be overwritten. The recomended file
        /// extension when saving data is `.mts`, to prevent confusion with generic
        /// `.npz` files.
        void save(const std::string& path, const TensorBlock& block);

        /// Save a `TensorBlock` to an in-memory buffer.
        ///
        /// The `Buffer` template parameter can be set to any type that can be
        /// constructed from a pair of iterator over `std::vector<uint8_t>`.
        template <typename Buffer = std::vector<uint8_t>>
        Buffer save_buffer(const TensorBlock& block);

        template<>
        std::vector<uint8_t> save_buffer<std::vector<uint8_t>>(const TensorBlock& block);

        /**************************************************************************/

        /// Save `Labels` to the file at `path`.
        ///
        /// If the file exists, it will be overwritten. The recomended file
        /// extension when saving data is `.mts`, to prevent confusion with generic
        /// `.npz` files.
        void save(const std::string& path, const Labels& labels);

        /// Save `Labels` to an in-memory buffer.
        ///
        /// The `Buffer` template parameter can be set to any type that can be
        /// constructed from a pair of iterator over `std::vector<uint8_t>`.
        template <typename Buffer = std::vector<uint8_t>>
        Buffer save_buffer(const Labels& labels);

        template<>
        std::vector<uint8_t> save_buffer<std::vector<uint8_t>>(const Labels& labels);

        /**************************************************************************/
        /**************************************************************************/

        /*!
        * Load a previously saved `TensorMap` from the given path.
        *
        * \verbatim embed:rst:leading-asterisk
        *
        * ``create_array`` will be used to create new arrays when constructing the
        * blocks and gradients, the default version will create data using
        * :cpp:class:`SimpleDataArray`. See :c:func:`mts_create_array_callback_t`
        * for more information.
        *
        * \endverbatim
        *
        * `TensorMap` are serialized using numpy's NPZ format, i.e. a ZIP file
        * without compression (storage method is `STORED`), where each file is
        * stored as a `.npy` array. See the C API documentation for more
        * information on the format.
        */
        TensorMap load(
            const std::string& path,
            mts_create_array_callback_t create_array = details::default_create_array
        );

        /*!
        * Load a previously saved `TensorMap` from the given `buffer`, containing
        * `buffer_count` elements.
        *
        * \verbatim embed:rst:leading-asterisk
        *
        * ``create_array`` will be used to create new arrays when constructing the
        * blocks and gradients, the default version will create data using
        * :cpp:class:`SimpleDataArray`. See :c:func:`mts_create_array_callback_t`
        * for more information.
        *
        * \endverbatim
        */
        TensorMap load_buffer(
            const uint8_t* buffer,
            size_t buffer_count,
            mts_create_array_callback_t create_array = details::default_create_array
        );


        /// Load a previously saved `TensorMap` from the given `buffer`.
        ///
        /// The `Buffer` template parameter would typically be a
        /// `std::vector<uint8_t>` or a `std::string`, but any container with
        /// contiguous data and an `item_type` with the same size as a `uint8_t` can
        /// work.
        template <typename Buffer>
        TensorMap load_buffer(
            const Buffer& buffer,
            mts_create_array_callback_t create_array = details::default_create_array
        );

        /**************************************************************************/

        /*!
        * Load a previously saved `TensorBlock` from the given path.
        *
        * \verbatim embed:rst:leading-asterisk
        *
        * ``create_array`` will be used to create new arrays when constructing the
        * blocks and gradients, the default version will create data using
        * :cpp:class:`SimpleDataArray`. See :c:func:`mts_create_array_callback_t`
        * for more information.
        *
        * \endverbatim
        *
        */
        TensorBlock load_block(
            const std::string& path,
            mts_create_array_callback_t create_array = details::default_create_array
        );

        /*!
        * Load a previously saved `TensorBlock` from the given `buffer`, containing
        * `buffer_count` elements.
        *
        * \verbatim embed:rst:leading-asterisk
        *
        * ``create_array`` will be used to create new arrays when constructing the
        * blocks and gradients, the default version will create data using
        * :cpp:class:`SimpleDataArray`. See :c:func:`mts_create_array_callback_t`
        * for more information.
        *
        * \endverbatim
        */
        TensorBlock load_block_buffer(
            const uint8_t* buffer,
            size_t buffer_count,
            mts_create_array_callback_t create_array = details::default_create_array
        );


        /// Load a previously saved `TensorBlock` from the given `buffer`.
        ///
        /// The `Buffer` template parameter would typically be a
        /// `std::vector<uint8_t>` or a `std::string`, but any container with
        /// contiguous data and an `item_type` with the same size as a `uint8_t` can
        /// work.
        template <typename Buffer>
        TensorBlock load_block_buffer(
            const Buffer& buffer,
            mts_create_array_callback_t create_array = details::default_create_array
        );

        /**************************************************************************/

        /// Load previously saved `Labels` from the given path.
        Labels load_labels(const std::string& path);

        /// Load previously saved `Labels` from the given `buffer`, containing
        /// `buffer_count` elements.
        Labels load_labels_buffer(const uint8_t* buffer, size_t buffer_count);

        /// Load a previously saved `Labels` from the given `buffer`.
        ///
        /// The `Buffer` template parameter would typically be a
        /// `std::vector<uint8_t>` or a `std::string`, but any container with
        /// contiguous data and an `item_type` with the same size as a `uint8_t` can
        /// work.
        template <typename Buffer>
        Labels load_labels_buffer(const Buffer& buffer);
    } // namespace io
} // namespace metatensor
