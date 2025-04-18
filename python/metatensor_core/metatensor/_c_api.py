# fmt: off
# flake8: noqa
"""
This file declares the C-API corresponding to metatensor.h, in a way compatible
with the ctypes Python module.

This file is automatically generated by `python/scripts/generate-declarations.py`,
do not edit it manually!
"""

import ctypes
import enum
import platform
from ctypes import CFUNCTYPE, POINTER


arch = platform.architecture()[0]
if arch == "32bit":
    c_uintptr_t = ctypes.c_uint32
elif arch == "64bit":
    c_uintptr_t = ctypes.c_uint64

MTS_SUCCESS = 0
MTS_INVALID_PARAMETER_ERROR = 1
MTS_IO_ERROR = 2
MTS_SERIALIZATION_ERROR = 3
MTS_BUFFER_SIZE_ERROR = 254
MTS_INTERNAL_ERROR = 255


mts_status_t = ctypes.c_int32
mts_data_origin_t = ctypes.c_uint64
mts_realloc_buffer_t = CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, c_uintptr_t)


class mts_block_t(ctypes.Structure):
    pass


class mts_tensormap_t(ctypes.Structure):
    pass


class mts_labels_t(ctypes.Structure):
    pass

mts_labels_t._fields_ = [
    ("internal_ptr_", ctypes.c_void_p),
    ("names", POINTER(ctypes.c_char_p)),
    ("values", POINTER(ctypes.c_int32)),
    ("size", c_uintptr_t),
    ("count", c_uintptr_t),
]


class mts_sample_mapping_t(ctypes.Structure):
    pass

mts_sample_mapping_t._fields_ = [
    ("input", c_uintptr_t),
    ("output", c_uintptr_t),
]


class mts_array_t(ctypes.Structure):
    pass

mts_array_t._fields_ = [
    ("ptr", ctypes.c_void_p),
    ("origin", CFUNCTYPE(mts_status_t, ctypes.c_void_p, POINTER(mts_data_origin_t))),
    ("data", CFUNCTYPE(mts_status_t, ctypes.c_void_p, POINTER(POINTER(ctypes.c_double)))),
    ("shape", CFUNCTYPE(mts_status_t, ctypes.c_void_p, POINTER(POINTER(c_uintptr_t)), POINTER(c_uintptr_t))),
    ("reshape", CFUNCTYPE(mts_status_t, ctypes.c_void_p, POINTER(c_uintptr_t), c_uintptr_t)),
    ("swap_axes", CFUNCTYPE(mts_status_t, ctypes.c_void_p, c_uintptr_t, c_uintptr_t)),
    ("create", CFUNCTYPE(mts_status_t, ctypes.c_void_p, POINTER(c_uintptr_t), c_uintptr_t, POINTER(mts_array_t))),
    ("copy", CFUNCTYPE(mts_status_t, ctypes.c_void_p, POINTER(mts_array_t))),
    ("destroy", CFUNCTYPE(None, ctypes.c_void_p)),
    ("move_samples_from", CFUNCTYPE(mts_status_t, ctypes.c_void_p, ctypes.c_void_p, POINTER(mts_sample_mapping_t), c_uintptr_t, c_uintptr_t, c_uintptr_t)),
]


mts_create_array_callback_t = CFUNCTYPE(mts_status_t, POINTER(c_uintptr_t), c_uintptr_t, POINTER(mts_array_t))


def setup_functions(lib):
    from .status import _check_status

    lib.mts_disable_panic_printing.argtypes = [
    ]
    lib.mts_disable_panic_printing.restype = None

    lib.mts_version.argtypes = [
    ]
    lib.mts_version.restype = ctypes.c_char_p

    lib.mts_last_error.argtypes = [
    ]
    lib.mts_last_error.restype = ctypes.c_char_p

    lib.mts_labels_position.argtypes = [
        mts_labels_t,
        POINTER(ctypes.c_int32),
        c_uintptr_t,
        POINTER(ctypes.c_int64),
    ]
    lib.mts_labels_position.restype = _check_status

    lib.mts_labels_create.argtypes = [
        POINTER(mts_labels_t),
    ]
    lib.mts_labels_create.restype = _check_status

    lib.mts_labels_set_user_data.argtypes = [
        mts_labels_t,
        ctypes.c_void_p,
        CFUNCTYPE(None, ctypes.c_void_p),
    ]
    lib.mts_labels_set_user_data.restype = _check_status

    lib.mts_labels_user_data.argtypes = [
        mts_labels_t,
        POINTER(POINTER(None)),
    ]
    lib.mts_labels_user_data.restype = _check_status

    lib.mts_labels_clone.argtypes = [
        mts_labels_t,
        POINTER(mts_labels_t),
    ]
    lib.mts_labels_clone.restype = _check_status

    lib.mts_labels_union.argtypes = [
        mts_labels_t,
        mts_labels_t,
        POINTER(mts_labels_t),
        POINTER(ctypes.c_int64),
        c_uintptr_t,
        POINTER(ctypes.c_int64),
        c_uintptr_t,
    ]
    lib.mts_labels_union.restype = _check_status

    lib.mts_labels_intersection.argtypes = [
        mts_labels_t,
        mts_labels_t,
        POINTER(mts_labels_t),
        POINTER(ctypes.c_int64),
        c_uintptr_t,
        POINTER(ctypes.c_int64),
        c_uintptr_t,
    ]
    lib.mts_labels_intersection.restype = _check_status

    lib.mts_labels_difference.argtypes = [
        mts_labels_t,
        mts_labels_t,
        POINTER(mts_labels_t),
        POINTER(ctypes.c_int64),
        c_uintptr_t,
    ]
    lib.mts_labels_difference.restype = _check_status

    lib.mts_labels_select.argtypes = [
        mts_labels_t,
        mts_labels_t,
        POINTER(ctypes.c_int64),
        POINTER(c_uintptr_t),
    ]
    lib.mts_labels_select.restype = _check_status

    lib.mts_labels_free.argtypes = [
        POINTER(mts_labels_t),
    ]
    lib.mts_labels_free.restype = _check_status

    lib.mts_register_data_origin.argtypes = [
        ctypes.c_char_p,
        POINTER(mts_data_origin_t),
    ]
    lib.mts_register_data_origin.restype = _check_status

    lib.mts_get_data_origin.argtypes = [
        mts_data_origin_t,
        ctypes.c_char_p,
        c_uintptr_t,
    ]
    lib.mts_get_data_origin.restype = _check_status

    lib.mts_block.argtypes = [
        mts_array_t,
        mts_labels_t,
        POINTER(mts_labels_t),
        c_uintptr_t,
        mts_labels_t,
    ]
    lib.mts_block.restype = POINTER(mts_block_t)

    lib.mts_block_free.argtypes = [
        POINTER(mts_block_t),
    ]
    lib.mts_block_free.restype = _check_status

    lib.mts_block_copy.argtypes = [
        POINTER(mts_block_t),
    ]
    lib.mts_block_copy.restype = POINTER(mts_block_t)

    lib.mts_block_labels.argtypes = [
        POINTER(mts_block_t),
        c_uintptr_t,
        POINTER(mts_labels_t),
    ]
    lib.mts_block_labels.restype = _check_status

    lib.mts_block_gradient.argtypes = [
        POINTER(mts_block_t),
        ctypes.c_char_p,
        POINTER(POINTER(mts_block_t)),
    ]
    lib.mts_block_gradient.restype = _check_status

    lib.mts_block_data.argtypes = [
        POINTER(mts_block_t),
        POINTER(mts_array_t),
    ]
    lib.mts_block_data.restype = _check_status

    lib.mts_block_add_gradient.argtypes = [
        POINTER(mts_block_t),
        ctypes.c_char_p,
        POINTER(mts_block_t),
    ]
    lib.mts_block_add_gradient.restype = _check_status

    lib.mts_block_gradients_list.argtypes = [
        POINTER(mts_block_t),
        POINTER(POINTER(ctypes.c_char_p)),
        POINTER(c_uintptr_t),
    ]
    lib.mts_block_gradients_list.restype = _check_status

    lib.mts_tensormap.argtypes = [
        mts_labels_t,
        POINTER(POINTER(mts_block_t)),
        c_uintptr_t,
    ]
    lib.mts_tensormap.restype = POINTER(mts_tensormap_t)

    lib.mts_tensormap_free.argtypes = [
        POINTER(mts_tensormap_t),
    ]
    lib.mts_tensormap_free.restype = _check_status

    lib.mts_tensormap_copy.argtypes = [
        POINTER(mts_tensormap_t),
    ]
    lib.mts_tensormap_copy.restype = POINTER(mts_tensormap_t)

    lib.mts_tensormap_keys.argtypes = [
        POINTER(mts_tensormap_t),
        POINTER(mts_labels_t),
    ]
    lib.mts_tensormap_keys.restype = _check_status

    lib.mts_tensormap_block_by_id.argtypes = [
        POINTER(mts_tensormap_t),
        POINTER(POINTER(mts_block_t)),
        c_uintptr_t,
    ]
    lib.mts_tensormap_block_by_id.restype = _check_status

    lib.mts_tensormap_blocks_matching.argtypes = [
        POINTER(mts_tensormap_t),
        POINTER(c_uintptr_t),
        POINTER(c_uintptr_t),
        mts_labels_t,
    ]
    lib.mts_tensormap_blocks_matching.restype = _check_status

    lib.mts_tensormap_keys_to_properties.argtypes = [
        POINTER(mts_tensormap_t),
        mts_labels_t,
        ctypes.c_bool,
    ]
    lib.mts_tensormap_keys_to_properties.restype = POINTER(mts_tensormap_t)

    lib.mts_tensormap_components_to_properties.argtypes = [
        POINTER(mts_tensormap_t),
        POINTER(ctypes.c_char_p),
        c_uintptr_t,
    ]
    lib.mts_tensormap_components_to_properties.restype = POINTER(mts_tensormap_t)

    lib.mts_tensormap_keys_to_samples.argtypes = [
        POINTER(mts_tensormap_t),
        mts_labels_t,
        ctypes.c_bool,
    ]
    lib.mts_tensormap_keys_to_samples.restype = POINTER(mts_tensormap_t)

    lib.mts_labels_load.argtypes = [
        ctypes.c_char_p,
        POINTER(mts_labels_t),
    ]
    lib.mts_labels_load.restype = _check_status

    lib.mts_labels_load_buffer.argtypes = [
        ctypes.c_char_p,
        c_uintptr_t,
        POINTER(mts_labels_t),
    ]
    lib.mts_labels_load_buffer.restype = _check_status

    lib.mts_labels_save.argtypes = [
        ctypes.c_char_p,
        mts_labels_t,
    ]
    lib.mts_labels_save.restype = _check_status

    lib.mts_labels_save_buffer.argtypes = [
        POINTER(ctypes.c_char_p),
        POINTER(c_uintptr_t),
        ctypes.c_void_p,
        mts_realloc_buffer_t,
        mts_labels_t,
    ]
    lib.mts_labels_save_buffer.restype = _check_status

    lib.mts_block_load.argtypes = [
        ctypes.c_char_p,
        mts_create_array_callback_t,
    ]
    lib.mts_block_load.restype = POINTER(mts_block_t)

    lib.mts_block_load_buffer.argtypes = [
        ctypes.c_char_p,
        c_uintptr_t,
        mts_create_array_callback_t,
    ]
    lib.mts_block_load_buffer.restype = POINTER(mts_block_t)

    lib.mts_block_save.argtypes = [
        ctypes.c_char_p,
        POINTER(mts_block_t),
    ]
    lib.mts_block_save.restype = _check_status

    lib.mts_block_save_buffer.argtypes = [
        POINTER(ctypes.c_char_p),
        POINTER(c_uintptr_t),
        ctypes.c_void_p,
        mts_realloc_buffer_t,
        POINTER(mts_block_t),
    ]
    lib.mts_block_save_buffer.restype = _check_status

    lib.mts_tensormap_load.argtypes = [
        ctypes.c_char_p,
        mts_create_array_callback_t,
    ]
    lib.mts_tensormap_load.restype = POINTER(mts_tensormap_t)

    lib.mts_tensormap_load_buffer.argtypes = [
        ctypes.c_char_p,
        c_uintptr_t,
        mts_create_array_callback_t,
    ]
    lib.mts_tensormap_load_buffer.restype = POINTER(mts_tensormap_t)

    lib.mts_tensormap_save.argtypes = [
        ctypes.c_char_p,
        POINTER(mts_tensormap_t),
    ]
    lib.mts_tensormap_save.restype = _check_status

    lib.mts_tensormap_save_buffer.argtypes = [
        POINTER(ctypes.c_char_p),
        POINTER(c_uintptr_t),
        ctypes.c_void_p,
        mts_realloc_buffer_t,
        POINTER(mts_tensormap_t),
    ]
    lib.mts_tensormap_save_buffer.restype = _check_status
