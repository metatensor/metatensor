#include <stdint.h>

#include <metatensor.h>


int main(void) {
    const char* names[1] = {"name"};
    int32_t values[3] = {1, 2, 3};

    mts_labels_t* labels = mts_labels_create(names, 1, values, 3);
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
