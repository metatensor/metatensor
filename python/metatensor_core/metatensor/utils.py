import os


try:
    from ._external import EXTERNAL_METATENSOR_PREFIX

    cmake_prefix_path = EXTERNAL_METATENSOR_PREFIX
    """
    Path containing the CMake configuration files for the underlying C library
    """

except ImportError:
    cmake_prefix_path = os.path.join(os.path.dirname(__file__), "lib", "cmake")
    """
    Path containing the CMake configuration files for the underlying C library
    """
