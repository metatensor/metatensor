# -*- coding: utf-8 -*-
'''
Automatically-generated file, do not edit!!!
'''
# flake8: noqa

import enum
import platform

import ctypes
from ctypes import POINTER, CFUNCTYPE
from numpy.ctypeslib import ndpointer

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


class aml_label_kind(enum.Enum):
    AML_SAMPLE_LABELS = 0
    AML_SYMMETRIC_LABELS = 1
    AML_FEATURE_LABELS = 2


class aml_block_t(ctypes.Structure):
    pass


class aml_descriptor_t(ctypes.Structure):
    pass


class aml_data_storage_t(ctypes.Structure):
    pass

aml_data_storage_t._fields_ = [
    ("data", ctypes.c_void_p),
    ("origin", CFUNCTYPE(aml_status_t, ctypes.c_void_p, POINTER(aml_data_origin_t))),
    ("set_from_other", CFUNCTYPE(aml_status_t, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)),
    ("reshape", CFUNCTYPE(aml_status_t, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)),
    ("create", CFUNCTYPE(aml_status_t, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, POINTER(aml_data_storage_t))),
    ("destroy", CFUNCTYPE(None, ctypes.c_void_p)),
]


class aml_labels_t(ctypes.Structure):
    pass

aml_labels_t._fields_ = [
    ("names", POINTER(ctypes.c_char_p)),
    ("values", POINTER(ctypes.c_int32)),
    ("size", c_uintptr_t),
    ("count", c_uintptr_t),
]


def setup_functions(lib):
    from .status import _check_status

    lib.aml_last_error.argtypes = [
        
    ]
    lib.aml_last_error.restype = ctypes.c_char_p

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
        aml_data_storage_t,
        aml_labels_t,
        aml_labels_t,
        aml_labels_t
    ]
    lib.aml_block.restype = POINTER(aml_block_t)

    lib.aml_block_free.argtypes = [
        POINTER(aml_block_t)
    ]
    lib.aml_block_free.restype = _check_status

    lib.aml_block_labels.argtypes = [
        POINTER(aml_block_t),
        ctypes.c_char_p,
        ctypes.c_int,
        POINTER(aml_labels_t)
    ]
    lib.aml_block_labels.restype = _check_status

    lib.aml_block_data.argtypes = [
        POINTER(aml_block_t),
        ctypes.c_char_p,
        POINTER(POINTER(aml_data_storage_t))
    ]
    lib.aml_block_data.restype = _check_status

    lib.aml_block_add_gradient.argtypes = [
        POINTER(aml_block_t),
        ctypes.c_char_p,
        aml_labels_t,
        aml_data_storage_t
    ]
    lib.aml_block_add_gradient.restype = _check_status

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

    lib.aml_descriptor_sparse_to_features.argtypes = [
        POINTER(aml_descriptor_t),
        POINTER(ctypes.c_char_p),
        ctypes.c_uint64
    ]
    lib.aml_descriptor_sparse_to_features.restype = _check_status

    lib.aml_descriptor_symmetric_to_features.argtypes = [
        POINTER(aml_descriptor_t)
    ]
    lib.aml_descriptor_symmetric_to_features.restype = _check_status

    lib.aml_descriptor_sparse_to_samples.argtypes = [
        POINTER(aml_descriptor_t),
        POINTER(ctypes.c_char_p),
        ctypes.c_uint64
    ]
    lib.aml_descriptor_sparse_to_samples.restype = _check_status
