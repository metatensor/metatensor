/* ============    Automatically generated file, DOT NOT EDIT.    ============ *
 *                                                                             *
 *    This file is automatically generated from the aml-storage sources,       *
 *    using cbindgen. If you want to make change to this file (including       *
 *    documentation), make the corresponding changes in the rust sources.      *
 * =========================================================================== */

#ifndef AML_STORAGE_H
#define AML_STORAGE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Status code used when a function succeeded
 */
#define AML_SUCCESS 0

/**
 * Status code used when a function got an invalid parameter
 */
#define AML_INVALID_PARAMETER_ERROR 1

/**
 * Status code used when a memory buffer is too small to fit the requested data
 */
#define AML_BUFFER_SIZE_ERROR 254

/**
 * Status code used when there was an internal error, i.e. there is a bug
 * inside AML itself
 */
#define AML_INTERNAL_ERROR 255

/**
 * The different kinds of labels that can exist on a `aml_block_t`
 */
typedef enum aml_label_kind {
  /**
   * The sample labels, describing different samples in the data
   */
  AML_SAMPLE_LABELS = 0,
  /**
   * The component labels, describing the components of vectorial or
   * tensorial elements of the data
   */
  AML_COMPONENTS_LABELS = 1,
  /**
   * The feature labels, describing the features of the data
   */
  AML_FEATURE_LABELS = 2,
} aml_label_kind;

/**
 * Basic building block for descriptor. A single block contains a 3-dimensional
 * `aml_array_t`, and three sets of `aml_labels_t` (one for each dimension).
 *
 * A block can also contain gradients of the values with respect to a variety
 * of parameters. In this case, each gradient has a separate set of samples,
 * but share the same components and feature labels as the values.
 */
typedef struct aml_block_t aml_block_t;

/**
 * Opaque type representing a `Descriptor`.
 */
typedef struct aml_descriptor_t aml_descriptor_t;

/**
 * Status type returned by all functions in the C API.
 *
 * The value 0 (`AML_SUCCESS`) is used to indicate successful operations,
 * positive values are used by this library to indicate errors, while negative
 * values are reserved for users of this library to indicate their own errors
 * in callbacks.
 */
typedef int32_t aml_status_t;

/**
 * A set of labels used to carry metadata associated with a descriptor.
 *
 * This is similar to a list of `n_entries` named tuples, but stored as a 2D
 * array of shape `(n_entries, n_variables)`, with a set of names associated
 * with the columns of this array (often called *variables*). Each row/entry in
 * this array is unique, and they are often (but not always) sorted in
 * lexicographic order.
 */
typedef struct aml_labels_t {
  /**
   * internal: pointer to the rust `Labels` struct if any, null otherwise
   */
  const void *labels_ptr;
  /**
   * Names of the variables composing this set of labels. There are `size`
   * elements in this array, each being a NULL terminated UTF-8 string.
   */
  const char *const *names;
  /**
   * Pointer to the first element of a 2D row-major array of 32-bit signed
   * integer containing the values taken by the different variables in
   * `names`. Each row has `size` elements, and there are `count` rows in
   * total.
   */
  const int32_t *values;
  /**
   * Number of variables/size of a single entry in the set of labels
   */
  uintptr_t size;
  /**
   * Number entries in the set of labels
   */
  uintptr_t count;
} aml_labels_t;

/**
 * A single 64-bit integer representing a data origin (numpy ndarray, rust
 * ndarray, torch tensor, fortran array, ...).
 */
typedef uint64_t aml_data_origin_t;

/**
 * `aml_array_t` manages 3D arrays the be used as data in a block/descriptor.
 * The array itself if opaque to this library and can come from multiple
 * sources: Rust program, a C/C++ program, a Fortran program, Python with numpy
 * or torch. The data does not have to live on CPU, or even on the same machine
 * where this code is executed.
 *
 * This struct contains a C-compatible manual implementation of a virtual table
 * (vtable, i.e. trait in Rust, pure virtual class in C++); allowing
 * manipulation of the array in an opaque way.
 */
typedef struct aml_array_t {
  /**
   * User-provided data should be stored here, it will be passed as the
   * first parameter to all function pointers below.
   */
  void *ptr;
  /**
   * This function needs to store the "data origin" for this array in
   * `origin`. Users of `aml_array_t` should register a single data
   * origin with `register_data_origin`, and use it for all compatible
   * arrays.
   */
  aml_status_t (*origin)(const void *array, aml_data_origin_t *origin);
  /**
   * Get the shape of the array managed by this `aml_array_t`
   */
  aml_status_t (*shape)(const void *array, uint64_t *n_samples, uint64_t *n_components, uint64_t *n_features);
  /**
   * Change the shape of the array managed by this `aml_array_t` to
   * `(n_samples, n_components, n_features)`
   */
  aml_status_t (*reshape)(void *array, uint64_t n_samples, uint64_t n_components, uint64_t n_features);
  /**
   * Create a new array with the same options as the current one (data type,
   * data location, etc.) and the requested `(n_samples, n_components,
   * n_features)` shape; and store it in `new_array`. The new array should be
   * filled with zeros.
   */
  aml_status_t (*create)(const void *array, uint64_t n_samples, uint64_t n_components, uint64_t n_features, struct aml_array_t *new_array);
  /**
   * Make a copy of this `array` and return the new array in `new_array`
   */
  aml_status_t (*copy)(const void *array, struct aml_array_t *new_array);
  /**
   * Set entries in this array taking data from the `other_array`. This array
   * is guaranteed to be created by calling `aml_array_t::create` with one of
   * the arrays in the same block or descriptor as this `array`.
   *
   * This function should copy data from `other_array[other_sample, :, :]` to
   * `array[sample, :, feature_start:feature_end]`. All indexes are 0-based.
   */
  aml_status_t (*move_sample)(void *array, uint64_t sample, uint64_t feature_start, uint64_t feature_end, const void *other_array, uint64_t other_sample);
  /**
   * Set entries in this array taking data from the `other_array`. This array
   * is guaranteed to be created by calling `aml_array_t::create` with one of
   * the arrays in the same block or descriptor as this `array`.
   *
   * This function should copy data from `other_array[:, other_component, :]`
   * to `array[:, component, feature_start:feature_end]`. All indexes are
   * 0-based.
   */
  aml_status_t (*move_component)(void *array, uint64_t component, uint64_t feature_start, uint64_t feature_end, const void *other_array, uint64_t other_component);
  /**
   * Remove this array and free the associated memory. This function can be
   * set to `NULL` is there is no memory management to do.
   */
  void (*destroy)(void *array);
} aml_array_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Get the last error message that was created on the current thread.
 *
 * @returns the last error message, as a NULL-terminated string
 */
const char *aml_last_error(void);

/**
 * Get the position of the entry defined by the `values` array in the given set
 * of `labels`. This operation is only available if the labels correspond to a
 * set of Rust Labels (i.e. `labels.labels_ptr` is not NULL).
 *
 * @param labels set of labels coming from an `aml_block_t` or an `aml_descriptor_t`
 * @param values array containing the label to lookup
 * @param count size of the values array
 * @param result position of the values in the labels or -1 if the values
 *               were not found
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_labels_position(struct aml_labels_t labels,
                                 const int32_t *values,
                                 uint64_t count,
                                 int64_t *result);

/**
 * Register a new data origin with the given `name`. Calling this function
 * multiple times with the same name will give the same `aml_data_origin_t`.
 *
 * @param name name of the data origin as an UTF-8 encoded NULL-terminated string
 * @param origin pointer to an `aml_data_origin_t` where the origin will be stored
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_register_data_origin(const char *name, aml_data_origin_t *origin);

/**
 * Get the name used to register a given data `origin` in the given `buffer`
 *
 * @param origin pre-registered data origin
 * @param buffer buffer to be filled with the data origin name. The origin name
 *               will be written  as an UTF-8 encoded, NULL-terminated string
 * @param buffer_size size of the buffer
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_get_data_origin(aml_data_origin_t origin, char *buffer, uint64_t buffer_size);

/**
 * Create a new `aml_block_t` with the given `data` and `samples`, `components`
 * and `features` labels.
 *
 * The memory allocated by this function and the blocks should be released
 * using `aml_block_free`, or moved into a descriptor using `aml_descriptor`.
 *
 * @param data array handle containing the data for this block. The block takes
 *             ownership of the array, and will release it with
 *             `array.destroy(array.ptr)` when it no longer needs it.
 * @param samples sample labels corresponding to the first dimension of the data
 * @param components component labels corresponding to the second dimension of the data
 * @param features feature labels corresponding to the third dimension of the data
 *
 * @returns A pointer to the newly allocated block, or a `NULL` pointer in
 *          case of error. In case of error, you can use `aml_last_error()`
 *          to get the error message.
 */
struct aml_block_t *aml_block(struct aml_array_t data,
                              struct aml_labels_t samples,
                              struct aml_labels_t components,
                              struct aml_labels_t features);

/**
 * Free the memory associated with a `block` previously created with
 * `aml_block`.
 *
 * If `block` is `NULL`, this function does nothing.
 *
 * @param block pointer to an existing block, or `NULL`
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_block_free(struct aml_block_t *block);

/**
 * Make a copy of an `aml_block_t`.
 *
 * The memory allocated by this function and the blocks should be released
 * using `aml_block_free`, or moved into a descriptor using `aml_descriptor`.
 *
 * @param block existing block to copy
 *
 * @returns A pointer to the newly allocated block, or a `NULL` pointer in
 *          case of error. In case of error, you can use `aml_last_error()`
 *          to get the error message.
 */
struct aml_block_t *aml_block_copy(const struct aml_block_t *block);

/**
 * Get the set of labels of the requested `kind` from this `block`.
 *
 * The `values_gradients` parameter controls whether this function looks up
 * labels for `"values"` or one of the gradients in this block.
 *
 * The resulting `labels.values` points inside memory owned by the block, and
 * as such is only valid until the block is destroyed with `aml_block_free`, or
 * the containing descriptor is modified with one of the
 * `aml_descriptor_sparse_to_xxx` function.
 *
 * @param block pointer to an existing block
 * @param values_gradients either `"values"` or the name of gradients to lookup
 * @param kind the kind of labels requested
 * @param labels pointer to an empty `aml_labels_t` that will be set to the
 *               requested labels
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_block_labels(const struct aml_block_t *block,
                              const char *values_gradients,
                              enum aml_label_kind kind,
                              struct aml_labels_t *labels);

/**
 * Get the array handle for either values or one of the gradient in this `block`.
 *
 * The `values_gradients` parameter controls whether this function looks up
 * labels for `"values"` or one of the gradients in this block.
 *
 * @param block pointer to an existing block
 * @param values_gradients either `"values"` or the name of gradients to lookup
 * @param data pointer to an empty `aml_array_t` that will be set to the
 *             requested array
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_block_data(const struct aml_block_t *block,
                            const char *values_gradients,
                            struct aml_array_t *data);

/**
 * Add a new gradient to this `block` with the given `name`.
 *
 * @param block pointer to an existing block
 * @param name name of the gradient as a NULL-terminated UTF-8 string. This is
 *             usually the parameter used when taking derivatives (e.g.
 *             `"positions"`, `"cell"`, etc.)
 * @param samples sample labels for the gradient array. The components and
 *                feature labels are supposed to match the values in this block
 * @param gradient array containing the gradient data. The block takes
 *                 ownership of the array, and will release it with
 *                 `array.destroy(array.ptr)` when it no longer needs it.
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_block_add_gradient(struct aml_block_t *block,
                                    const char *name,
                                    struct aml_labels_t samples,
                                    struct aml_array_t gradient);

/**
 * Get a list of all gradients defined in this `block` in the `parameters` array.
 *
 * @param block pointer to an existing block
 * @param parameter will be set to the first element of an array of
 *                  NULL-terminated UTF-8 strings containing all the parameters
 *                  for which a gradient exists
 * @param count will be set to the number of elements in `parameters`
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_block_gradients_list(struct aml_block_t *block,
                                      const char *const **parameters,
                                      uint64_t *count);

/**
 * Create a new `aml_descriptor_t` with the given `sparse` labels and `blocks`.
 * `blocks_count` must be set to the number of entries in the blocks array.
 *
 * The new descriptor takes ownership of the blocks, which should not be
 * released separately.
 *
 * The memory allocated by this function and the blocks should be released
 * using `aml_descriptor_free`.
 *
 * @param sparse sparse labels associated with each block
 * @param blocks pointer to the first element of an array of blocks
 * @param blocks_count number of elements in the `blocks` array
 *
 * @returns A pointer to the newly allocated descriptor, or a `NULL` pointer in
 *          case of error. In case of error, you can use `aml_last_error()`
 *          to get the error message.
 */
struct aml_descriptor_t *aml_descriptor(struct aml_labels_t sparse,
                                        struct aml_block_t **blocks,
                                        uint64_t blocks_count);

/**
 * Free the memory associated with a `descriptor` previously created with
 * `aml_descriptor`.
 *
 * If `descriptor` is `NULL`, this function does nothing.
 *
 * @param descriptor pointer to an existing descriptor, or `NULL`
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_descriptor_free(struct aml_descriptor_t *descriptor);

/**
 * Get the sparse `labels` for the given `descriptor`. After a sucessful call
 * to this function, `labels.values` contains a pointer to memory inside the
 * `descriptor` which is invalidated when the descriptor is freed with
 * `aml_descriptor_free` or the set of sparse labels is modified by calling one
 * of the `aml_descriptor_sparse_to_XXX` function.
 *
 * @param descriptor pointer to an existing descriptor
 * @param labels pointer to be filled with the sparse labels of the descriptor
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_descriptor_sparse_labels(const struct aml_descriptor_t *descriptor,
                                          struct aml_labels_t *labels);

/**
 * Get a pointer to the `index`-th block in this descriptor.
 *
 * The block memory is still managed by the descriptor, this block should not
 * be freed. The block is invalidated when the descriptor is freed with
 * `aml_descriptor_free` or the set of sparse labels is modified by calling one
 * of the `aml_descriptor_sparse_to_XXX` function.
 *
 * @param descriptor pointer to an existing descriptor
 * @param block pointer to be filled with a block
 * @param index index of the block to get
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_descriptor_block_by_id(const struct aml_descriptor_t *descriptor,
                                        const struct aml_block_t **block,
                                        uint64_t index);

/**
 * Get a pointer to the `block` in this `descriptor` corresponding to the given
 * `selection`. The `selection` should have the same names/variables as the
 * sparse labels for this descriptor, and only one entry, describing the
 * requested block.
 *
 * The block memory is still managed by the descriptor, this block should not
 * be freed. The block is invalidated when the descriptor is freed with
 * `aml_descriptor_free` or the set of sparse labels is modified by calling one
 * of the `aml_descriptor_sparse_to_XXX` function.
 *
 * @param descriptor pointer to an existing descriptor
 * @param block pointer to be filled with a block
 * @param selection labels with a single entry describing which block is requested
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_descriptor_block_selection(const struct aml_descriptor_t *descriptor,
                                            const struct aml_block_t **block,
                                            struct aml_labels_t selection);

/**
 * Move the given variables from the sparse labels to the feature labels of the
 * blocks.
 *
 * The current blocks will be merged together according to the sparse labels
 * remaining after removing `variables`. The resulting merged blocks will have
 * `variables` as the first feature variables, followed by the current
 * features. The new sample labels will contains all of the merged blocks
 * sample labels, re-ordered to keep them lexicographically sorted.
 *
 * `variables` must be an array of `variables_count` NULL-terminated strings,
 * encoded as UTF-8.
 *
 * @param descriptor pointer to an existing descriptor
 * @param variables name of the sparse variables to move to the features
 * @param variables_count number of entries in the `variables` array
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_descriptor_sparse_to_features(struct aml_descriptor_t *descriptor,
                                               const char *const *variables,
                                               uint64_t variables_count);

/**
 * Move the given variables from the component labels to the feature labels for
 * each block in this descriptor.
 *
 * `variables` must be an array of `variables_count` NULL-terminated strings,
 * encoded as UTF-8.
 *
 * @param descriptor pointer to an existing descriptor
 * @param variables name of the sparse variables to move to the features
 * @param variables_count number of entries in the `variables` array
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_descriptor_components_to_features(struct aml_descriptor_t *descriptor,
                                                   const char *const *variables,
                                                   uint64_t variables_count);

/**
 * Move the given variables from the sparse labels to the sample labels of the
 * blocks.
 *
 * The current blocks will be merged together according to the sparse
 * labels remaining after removing `variables`. The resulting merged
 * blocks will have `variables` as the last sample variables, preceded by
 * the current samples.
 *
 * `variables` must be an array of `variables_count` NULL-terminated strings,
 * encoded as UTF-8.
 *
 * @param descriptor pointer to an existing descriptor
 * @param variables name of the sparse variables to move to the samples
 * @param variables_count number of entries in the `variables` array
 *
 * @returns The status code of this operation. If the status is not
 *          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
 *          error message.
 */
aml_status_t aml_descriptor_sparse_to_samples(struct aml_descriptor_t *descriptor,
                                              const char *const *variables,
                                              uint64_t variables_count);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* AML_STORAGE_H */
