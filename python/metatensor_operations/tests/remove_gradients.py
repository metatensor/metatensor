import os

import metatensor as mts

from . import _tests_utils


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_remove_everything():
    tensor = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))

    assert set(tensor.block(0).gradients_list()) == set(["strain", "positions"])

    tensor = mts.remove_gradients(tensor)
    assert tensor.block(0).gradients_list() == []


def test_remove_subset():
    tensor = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))

    assert set(tensor.block(0).gradients_list()) == set(["strain", "positions"])

    tensor = mts.remove_gradients(tensor, ["positions"])
    assert tensor.block(0).gradients_list() == ["strain"]


def test_remove_gradients_info():
    t = _tests_utils.simple_tensor()
    result = mts.remove_gradients(t)
    _tests_utils.check_info(result, {"description": "test_value"})
