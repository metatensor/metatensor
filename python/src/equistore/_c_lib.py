import os
import sys
from ctypes import cdll

from pkg_resources import parse_version

from ._c_api import setup_functions
from .data.extract import ExternalCpuArray, register_external_data_wrapper
from .version import __version__


_HERE = os.path.realpath(os.path.dirname(__file__))


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
            path = _lib_path()
            self._cached_dll = cdll.LoadLibrary(path)
            setup_functions(self._cached_dll)

            # initial setup, disable printing of the error in case of panic
            # the error will be transformed to a Python exception anyway
            self._cached_dll.eqs_disable_panic_printing()

            version = self._cached_dll.eqs_version().decode("utf8")
            if not _compatible_versions(version, __version__):
                self._cached_dll = None
                raise RuntimeError(
                    f"wrong version for libequistore, we want {__version__}, "
                    f"but we got {version} in '{path}'"
                )

            # Register the origin used by the Rust API as an external CPU array
            register_external_data_wrapper("rust.Box<dyn Array>", ExternalCpuArray)

        return self._cached_dll


def _lib_path():
    if sys.platform.startswith("darwin"):
        windows = False
        name = "libequistore.dylib"
    elif sys.platform.startswith("linux"):
        windows = False
        name = "libequistore.so"
    elif sys.platform.startswith("win"):
        windows = True
        name = "equistore.dll"
    else:
        raise ImportError("Unknown platform. Please edit this file")

    path = os.path.join(os.path.join(_HERE, "lib"), name)

    if os.path.isfile(path):
        if windows:
            _check_dll(path)
        return path

    raise ImportError("Could not find equistore shared library at " + path)


def _check_dll(path):
    """
    Check if the DLL pointer size matches Python (32-bit or 64-bit)
    """
    import platform
    import struct

    IMAGE_FILE_MACHINE_I386 = 332
    IMAGE_FILE_MACHINE_AMD64 = 34404

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

    arch = platform.architecture()[0]
    if arch == "32bit":
        if machine != IMAGE_FILE_MACHINE_I386:
            raise ImportError("Python is 32-bit, but this DLL is not")
    elif arch == "64bit":
        if machine != IMAGE_FILE_MACHINE_AMD64:
            raise ImportError("Python is 64-bit, but this DLL is not")
    else:
        raise ImportError("Could not determine pointer size of Python")


_get_library = LibraryFinder()
