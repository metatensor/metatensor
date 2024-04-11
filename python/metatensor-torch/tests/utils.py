import os

import metatensor.torch


def test_cmake_prefix_path():
    assert os.path.exists(metatensor.torch.utils.cmake_prefix_path)
