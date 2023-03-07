#include <stdint.h>

#include <equistore.h>


int main(void) {
    eqs_labels_t labels = {0};
    const char* names[1] = {"name"};
    labels.names = names;
    labels.size = 1;

    int32_t values[3] = {1, 2, 3};
    labels.values = values;
    labels.count = 3;

    eqs_status_t status = eqs_labels_create(&labels);
    if (status != EQS_SUCCESS) {
        goto fail;
    }

    status = eqs_labels_free(&labels);
    if (status != EQS_SUCCESS) {
        goto fail;
    }

    return 0;

fail:
    return 1;
}
