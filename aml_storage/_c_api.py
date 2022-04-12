# -*- coding: utf-8 -*-
'''
Automatically-generated file, do not edit!!!
'''
# flake8: noqa

import enum
import platform

import ctypes
from ctypes import POINTER, CFUNCTYPE

arch = platform.architecture()[0]
if arch == "32bit":
    c_uintptr_t = ctypes.c_uint32
elif arch == "64bit":
    c_uintptr_t = ctypes.c_uint64

AML_SUCCESS = 0
AML_INVALID_PARAMETER_ERROR = 1
AML_BUFFER_SIZE_ERROR = 254
AML_INTERNAL_ERROR = 255


aml_status_t = ctypes.c_int32
aml_data_origin_t = ctypes.c_uint64


class aml_block_t(ctypes.Structure):
    pass


class aml_descriptor_t(ctypes.Structure):
    pass


class aml_labels_t(ctypes.Structure):
    pass

aml_labels_t._fields_ = [
    ("labels_ptr", ctypes.c_void_p),
    ("names", POINTER(ctypes.c_char_p)),
    ("values", POINTER(ctypes.c_int32)),
    ("size", c_uintptr_t),
    ("count", c_uintptr_t),
]


class aml_array_t(ctypes.Structure):
    pass

aml_array_t._fields_ = [
    ("ptr", ctypes.c_void_p),
    ("origin", CFUNCTYPE(aml_status_t, ctypes.c_void_p, POINTER(aml_data_origin_t))),
    ("shape", CFUNCTYPE(aml_status_t, ctypes.c_void_p, POINTER(POINTER(c_uintptr_t)), POINTER(c_uintptr_t))),
    ("reshape", CFUNCTYPE(aml_status_t, ctypes.c_void_p, POINTER(c_uintptr_t), c_uintptr_t)),
    ("swap_axes", CFUNCTYPE(aml_status_t, ctypes.c_void_p, c_uintptr_t, c_uintptr_t)),
    ("create", CFUNCTYPE(aml_status_t, ctypes.c_void_p, POINTER(c_uintptr_t), c_uintptr_t, POINTER(aml_array_t))),
    ("copy", CFUNCTYPE(aml_status_t, ctypes.c_void_p, POINTER(aml_array_t))),
    ("destroy", CFUNCTYPE(None, ctypes.c_void_p)),
    ("move_sample", CFUNCTYPE(aml_status_t, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)),
]


def setup_functions(lib):
    from .status import _check_status

    lib.aml_last_error.argtypes = [
        
    ]
    lib.aml_last_error.restype = ctypes.c_char_p

    lib.aml_labels_position.argtypes = [
        aml_labels_t,
        POINTER(ctypes.c_int32),
        ctypes.c_uint64,
        POINTER(ctypes.c_int64)
    ]
    lib.aml_labels_position.restype = _check_status

    lib.aml_register_data_origin.argtypes = [
        ctypes.c_char_p,
        POINTER(aml_data_origin_t)
    ]
    lib.aml_register_data_origin.restype = _check_status

    lib.aml_get_data_origin.argtypes = [
        aml_data_origin_t,
        ctypes.c_char_p,
        ctypes.c_uint64
    ]
    lib.aml_get_data_origin.restype = _check_status

    lib.aml_block.argtypes = [
        aml_array_t,
        aml_labels_t,
        POINTER(aml_labels_t),
        c_uintptr_t,
        aml_labels_t
    ]
    lib.aml_block.restype = POINTER(aml_block_t)

    lib.aml_block_free.argtypes = [
        POINTER(aml_block_t)
    ]
    lib.aml_block_free.restype = _check_status

    lib.aml_block_copy.argtypes = [
        POINTER(aml_block_t)
    ]
    lib.aml_block_copy.restype = POINTER(aml_block_t)

    lib.aml_block_labels.argtypes = [
        POINTER(aml_block_t),
        ctypes.c_char_p,
        c_uintptr_t,
        POINTER(aml_labels_t)
    ]
    lib.aml_block_labels.restype = _check_status

    lib.aml_block_data.argtypes = [
        POINTER(aml_block_t),
        ctypes.c_char_p,
        POINTER(aml_array_t)
    ]
    lib.aml_block_data.restype = _check_status

    lib.aml_block_add_gradient.argtypes = [
        POINTER(aml_block_t),
        ctypes.c_char_p,
        aml_array_t,
        aml_labels_t,
        POINTER(aml_labels_t),
        c_uintptr_t
    ]
    lib.aml_block_add_gradient.restype = _check_status

    lib.aml_block_gradients_list.argtypes = [
        POINTER(aml_block_t),
        POINTER(POINTER(ctypes.c_char_p)),
        POINTER(ctypes.c_uint64)
    ]
    lib.aml_block_gradients_list.restype = _check_status

    lib.aml_descriptor.argtypes = [
        aml_labels_t,
        POINTER(POINTER(aml_block_t)),
        ctypes.c_uint64
    ]
    lib.aml_descriptor.restype = POINTER(aml_descriptor_t)

    lib.aml_descriptor_free.argtypes = [
        POINTER(aml_descriptor_t)
    ]
    lib.aml_descriptor_free.restype = _check_status

    lib.aml_descriptor_sparse_labels.argtypes = [
        POINTER(aml_descriptor_t),
        POINTER(aml_labels_t)
    ]
    lib.aml_descriptor_sparse_labels.restype = _check_status

    lib.aml_descriptor_block_by_id.argtypes = [
        POINTER(aml_descriptor_t),
        POINTER(POINTER(aml_block_t)),
        ctypes.c_uint64
    ]
    lib.aml_descriptor_block_by_id.restype = _check_status

    lib.aml_descriptor_block_selection.argtypes = [
        POINTER(aml_descriptor_t),
        POINTER(POINTER(aml_block_t)),
        aml_labels_t
    ]
    lib.aml_descriptor_block_selection.restype = _check_status

    lib.aml_descriptor_sparse_to_properties.argtypes = [
        POINTER(aml_descriptor_t),
        POINTER(ctypes.c_char_p),
        ctypes.c_uint64
    ]
    lib.aml_descriptor_sparse_to_properties.restype = _check_status

    lib.aml_descriptor_components_to_properties.argtypes = [
        POINTER(aml_descriptor_t),
        POINTER(ctypes.c_char_p),
        ctypes.c_uint64
    ]
    lib.aml_descriptor_components_to_properties.restype = _check_status

    lib.aml_descriptor_sparse_to_samples.argtypes = [
        POINTER(aml_descriptor_t),
        POINTER(ctypes.c_char_p),
        ctypes.c_uint64
    ]
    lib.aml_descriptor_sparse_to_samples.restype = _check_status
