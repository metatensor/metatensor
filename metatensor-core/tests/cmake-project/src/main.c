#include <stdint.h>

#include <metatensor.h>


int main(void) {
    mts_labels_t labels = {0};
    const char* names[1] = {"name"};
    labels.names = names;
    labels.size = 1;

    int32_t values[3] = {1, 2, 3};
    labels.values = values;
    labels.count = 3;

    mts_status_t status = mts_labels_create(&labels);
    if (status != MTS_SUCCESS) {
        goto fail;
    }

    status = mts_labels_free(&labels);
    if (status != MTS_SUCCESS) {
        goto fail;
    }

    return 0;

fail:
    return 1;
}
