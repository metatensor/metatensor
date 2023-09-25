import os
import re
import sys
from collections import namedtuple
from ctypes import cdll

from ._c_api import setup_functions
from .data.extract import ExternalCpuArray, register_external_data_wrapper
from .version import __version__


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
            path = _lib_path()
            self._cached_dll = cdll.LoadLibrary(path)
            setup_functions(self._cached_dll)

            # initial setup, disable printing of the error in case of panic
            # the error will be transformed to a Python exception anyway
            self._cached_dll.mts_disable_panic_printing()

            version = self._cached_dll.mts_version().decode("utf8")
            if not _compatible_versions(version, __version__):
                self._cached_dll = None
                raise RuntimeError(
                    f"wrong version for libmetatensor, we want {__version__}, "
                    f"but we got {version} @ '{path}'"
                )

            # Register the origin used by the Rust API as an external CPU array
            register_external_data_wrapper("rust.Box<dyn Array>", ExternalCpuArray)

        return self._cached_dll


def _lib_path():
    if sys.platform.startswith("darwin"):
        windows = False
        path = os.path.join(_HERE, "lib", "libmetatensor.dylib")
    elif sys.platform.startswith("linux"):
        windows = False
        path = os.path.join(_HERE, "lib", "libmetatensor.so")
    elif sys.platform.startswith("win"):
        windows = True
        path = os.path.join(_HERE, "bin", "metatensor.dll")
    else:
        raise ImportError("Unknown platform. Please edit this file")

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
