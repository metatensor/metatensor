import os

import metatensor as mts


def test_cmake_prefix_path():
    assert os.path.exists(mts.utils.cmake_prefix_path)
