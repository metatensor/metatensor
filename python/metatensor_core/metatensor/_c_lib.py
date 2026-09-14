import ctypes
import os
import re
import sys
from collections import namedtuple
from ctypes import cdll, wintypes

from ._c_api import setup_functions
from ._data._extract import ExternalCpuArray, register_external_data_wrapper
from ._version import __version__


_HERE = os.path.realpath(os.path.dirname(__file__))

Version = namedtuple("Version", ["major", "minor", "patch"])


def parse_version(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    if match:
        return Version(*map(int, match.groups()))
    else:
        raise ValueError("Invalid version string format")


def _compatible_versions(actual, minimal):
    actual = parse_version(actual)
    minimal = parse_version(minimal)

    # Different major version are not compatible
    if actual.major != minimal.major:
        return False

    # If the major version is 0, different minor version are not compatible
    if actual.major == 0 and actual.minor != minimal.minor:
        return False

    return True


class LibraryFinder(object):
    def __init__(self):
        self._cached_dll = None

    def __call__(self):
        if self._cached_dll is None:
            # if the library is already loaded in the current process, use this one
            # instead of loading a second, independent copy of it
            dll = _already_loaded(_lib_name())
            if dll is None:
                path = _lib_path()
                dll = cdll.LoadLibrary(path)
            else:
                path = "<already loaded in the current process>"

            self._cached_dll = dll
            setup_functions(self._cached_dll)

            # initial setup, disable printing of the error in case of panic
            # the error will be transformed to a Python exception anyway
            self._cached_dll.mts_disable_panic_printing()

            version = self._cached_dll.mts_version().decode("utf8")
            if not _compatible_versions(version, __version__):
                self._cached_dll = None
                raise RuntimeError(
                    f"wrong version for libmetatensor, we want v{__version__}, "
                    f"but we got v{version} @ '{path}'"
                )

            # Register the origin used by the Rust API as an external CPU array
            register_external_data_wrapper("RustArray", ExternalCpuArray)
            register_external_data_wrapper("metatensor.Labels", ExternalCpuArray)

        return self._cached_dll


def _lib_name():
    """Name of the metatensor shared library on the current platform"""
    if sys.platform.startswith("darwin"):
        return "libmetatensor.dylib"
    elif sys.platform.startswith("linux"):
        return "libmetatensor.so"
    elif sys.platform.startswith("win"):
        return "metatensor.dll"
    else:
        raise ImportError("Unknown platform. Please edit this file")


def _already_loaded(name):
    """
    Check if the library with the given ``name`` is already loaded in the current
    process, and return the corresponding ``CDLL`` if it is. This makes sure we share
    the global state of the library (error buffers, registered data origins, ...) with
    whoever loaded it first, instead of using a second, independent copy of the library.

    Returns ``None`` if the library is not already loaded.
    """
    if sys.platform.startswith("win"):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]

        handle = kernel32.GetModuleHandleW(name)
        if not handle:
            return None

        return ctypes.CDLL(name, handle=handle)
    else:
        # RTLD_NOLOAD gives us a handle if the library is already loaded, and fails
        # instead of loading it otherwise.
        RTLD_NOLOAD = getattr(os, "RTLD_NOLOAD", None)
        if RTLD_NOLOAD is None:
            return None

        try:
            return ctypes.CDLL(name, mode=RTLD_NOLOAD | os.RTLD_LOCAL)
        except OSError:
            return None


def _lib_path():
    try:
        # check if we are using an externally-provided version of the shared library
        from ._external import EXTERNAL_METATENSOR_PATH

        return EXTERNAL_METATENSOR_PATH
    except ImportError:
        pass

    # otherwise load from the local installation
    windows = sys.platform.startswith("win")
    if windows:
        path = os.path.join(_HERE, "bin", _lib_name())
    else:
        path = os.path.join(_HERE, "lib", _lib_name())

    if os.path.isfile(path):
        if windows:
            _check_dll(path)
        return path

    raise ImportError("Could not find metatensor shared library at " + path)


def _check_dll(path):
    """Check if the DLL at ``path`` matches the architecture of Python"""
    import platform
    import struct

    IMAGE_FILE_MACHINE_I386 = 332
    IMAGE_FILE_MACHINE_AMD64 = 34404
    IMAGE_FILE_MACHINE_ARM64 = 43620

    machine = None
    with open(path, "rb") as fd:
        header = fd.read(2).decode(encoding="utf-8", errors="strict")
        if header != "MZ":
            raise ImportError(path + " is not a DLL")
        else:
            fd.seek(60)
            header = fd.read(4)
            header_offset = struct.unpack("<L", header)[0]
            fd.seek(header_offset + 4)
            header = fd.read(2)
            machine = struct.unpack("<H", header)[0]

    python_machine = platform.machine()
    if python_machine == "x86":
        if machine != IMAGE_FILE_MACHINE_I386:
            raise ImportError("Python is 32-bit x86, but metatensor.dll is not")
    elif python_machine == "AMD64":
        if machine != IMAGE_FILE_MACHINE_AMD64:
            raise ImportError("Python is 64-bit x86_64, but metatensor.dll is not")
    elif python_machine == "ARM64":
        if machine != IMAGE_FILE_MACHINE_ARM64:
            raise ImportError("Python is 64-bit ARM, but metatensor.dll is not")
    else:
        raise ImportError(
            f"Metatensor doesn't provide a version for {python_machine} CPU. "
            "If you are compiling from source on a new architecture, edit this file"
        )


_get_library = LibraryFinder()
