import os

import metatensor


def test_cmake_prefix_path():
    assert os.path.exists(metatensor.utils.cmake_prefix_path)
