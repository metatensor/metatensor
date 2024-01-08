import os
import re
import sys
from collections import namedtuple

import torch

import metatensor

from ._build_versions import BUILD_METATENSOR_CORE_VERSION, BUILD_TORCH_VERSION
from .version import __version__


Version = namedtuple("Version", ["major", "minor", "patch"])


def parse_version(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    if match:
        return Version(*map(int, match.groups()))
    else:
        raise ValueError("Invalid version string format")


def version_compatible(actual, required):
    actual = parse_version(actual)
    required = parse_version(required)

    if actual.major != required.major:
        return False
    elif actual.minor != required.minor:
        return False
    else:
        return True


if not version_compatible(torch.__version__, BUILD_TORCH_VERSION):
    raise ImportError(
        f"Trying to load metatensor-torch with torch v{torch.__version__}, "
        f"but it was compiled against torch v{BUILD_TORCH_VERSION}, which "
        "is not ABI compatible"
    )

if not version_compatible(metatensor.__version__, BUILD_METATENSOR_CORE_VERSION):
    raise ImportError(
        "Trying to load metatensor-torch with metatensor-core "
        f"v{metatensor.__version__}, but it was compiled against "
        f"metatensor-core v{BUILD_METATENSOR_CORE_VERSION}, which "
        "is not ABI compatible"
    )

_HERE = os.path.realpath(os.path.dirname(__file__))


def _lib_path():
    if sys.platform.startswith("darwin"):
        path = os.path.join(_HERE, "lib", "libmetatensor_torch.dylib")
        windows = False
    elif sys.platform.startswith("linux"):
        path = os.path.join(_HERE, "lib", "libmetatensor_torch.so")
        windows = False
    elif sys.platform.startswith("win"):
        path = os.path.join(_HERE, "bin", "metatensor_torch.dll")
        windows = True
    else:
        raise ImportError("Unknown platform. Please edit this file")

    if os.path.isfile(path):
        if windows:
            _check_dll(path)
        return path

    raise ImportError("Could not find metatensor_torch shared library at " + path)


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


def _load_library():
    # Load metatensor shared library in the process first, to ensure
    # the metatensor_torch shared library can find it
    metatensor._c_lib._get_library()

    # load the C++ operators and custom classes
    torch.ops.load_library(_lib_path())

    lib_version = torch.ops.metatensor.version()
    if not version_compatible(lib_version, __version__):
        raise ImportError(
            f"Trying to load the Python package metatensor-torch v{__version__} "
            f"with the incompatible metatensor-torch C++ library v{lib_version}"
        )
