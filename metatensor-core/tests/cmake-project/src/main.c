#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <metatensor/dlpack/dlpack.h>
#include <metatensor.h>

/* Minimal mts_array_t implementation for labels test values */

typedef struct {
    const int32_t* data;
    uintptr_t shape[2];
} labels_test_array_t;

static mts_status_t test_origin(const void* array, mts_data_origin_t* origin) {
    (void)array;
    mts_register_data_origin("test_labels_array", origin);
    return MTS_SUCCESS;
}

static mts_status_t test_device(const void* array, DLDevice* device) {
    (void)array;
    device->device_type = kDLCPU;
    device->device_id = 0;
    return MTS_SUCCESS;
}

static mts_status_t test_dtype(const void* array, DLDataType* dtype) {
    (void)array;
    dtype->code = kDLInt;
    dtype->bits = 32;
    dtype->lanes = 1;
    return MTS_SUCCESS;
}

static mts_status_t test_shape(const void* array, const uintptr_t** shape, uintptr_t* shape_count) {
    const labels_test_array_t* a = (const labels_test_array_t*)array;
    *shape = a->shape;
    *shape_count = 2;
    return MTS_SUCCESS;
}

static void test_destroy(void* array) {
    free(array);
}

static mts_status_t test_copy(const void* array, mts_array_t* new_array) {
    /* Labels::from_array needs copy for CPU materialization */
    const labels_test_array_t* src = (const labels_test_array_t*)array;
    labels_test_array_t* dst = (labels_test_array_t*)malloc(sizeof(labels_test_array_t));
    if (dst == NULL) return -1;
    *dst = *src;

    memset(new_array, 0, sizeof(mts_array_t));
    new_array->ptr = dst;
    new_array->origin = test_origin;
    new_array->device = test_device;
    new_array->dtype = test_dtype;
    new_array->shape = test_shape;
    new_array->copy = test_copy;
    new_array->destroy = test_destroy;

    return MTS_SUCCESS;
}

static void test_dlpack_deleter(struct DLManagedTensorVersioned* self) {
    free(self->dl_tensor.shape);
    free(self);
}

static mts_status_t test_as_dlpack(
    void* array,
    DLManagedTensorVersioned** dl_managed_tensor_ptr,
    DLDevice device,
    const int64_t* stream,
    DLPackVersion max_version
) {
    const labels_test_array_t* a = (const labels_test_array_t*)array;

    /* Allocate a DLManagedTensorVersioned */
    DLManagedTensorVersioned* tensor = (DLManagedTensorVersioned*)calloc(1, sizeof(*tensor));
    if (tensor == NULL) return -1;

    tensor->version.major = 1;
    tensor->version.minor = 0;

    tensor->dl_tensor.data = (void*)a->data;
    tensor->dl_tensor.device.device_type = kDLCPU;
    tensor->dl_tensor.device.device_id = 0;
    tensor->dl_tensor.ndim = 2;
    tensor->dl_tensor.dtype.code = kDLInt;
    tensor->dl_tensor.dtype.bits = 32;
    tensor->dl_tensor.dtype.lanes = 1;

    /* shape array: allocate and copy */
    int64_t* shape = (int64_t*)malloc(2 * sizeof(int64_t));
    shape[0] = (int64_t)a->shape[0];
    shape[1] = (int64_t)a->shape[1];
    tensor->dl_tensor.shape = shape;
    tensor->dl_tensor.strides = NULL;
    tensor->dl_tensor.byte_offset = 0;

    tensor->deleter = test_dlpack_deleter;

    *dl_managed_tensor_ptr = tensor;
    (void)device;
    (void)stream;
    (void)max_version;
    return MTS_SUCCESS;
}

static mts_array_t make_labels_array(const int32_t* data, uintptr_t count, uintptr_t size) {
    labels_test_array_t* a = (labels_test_array_t*)malloc(sizeof(labels_test_array_t));
    a->data = data;
    a->shape[0] = count;
    a->shape[1] = size;

    mts_array_t array;
    memset(&array, 0, sizeof(array));
    array.ptr = a;
    array.origin = test_origin;
    array.device = test_device;
    array.dtype = test_dtype;
    array.shape = test_shape;
    array.copy = test_copy;
    array.destroy = test_destroy;
    array.as_dlpack = test_as_dlpack;

    return array;
}


int main(void) {
    const char* names[1] = {"name"};
    int32_t values[3] = {1, 2, 3};

    mts_array_t array = make_labels_array(values, 3, 1);
    mts_labels_t* labels = mts_labels_create(names, 1, array);
    if (labels == NULL) {
        goto fail;
    }

    mts_status_t status = mts_labels_free(labels);
    if (status != MTS_SUCCESS) {
        goto fail;
    }

    return 0;

fail:
    return 1;
}
