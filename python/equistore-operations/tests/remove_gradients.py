import os

import equistore


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_remove_everything():
    tensor = equistore.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        # the npz is using DEFLATE compression, equistore only supports STORED
        use_numpy=True,
    )

    assert set(tensor.block(0).gradients_list()) == set(["cell", "positions"])

    tensor = equistore.remove_gradients(tensor)
    assert tensor.block(0).gradients_list() == []


def test_remove_subset():
    tensor = equistore.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        # the npz is using DEFLATE compression, equistore only supports STORED
        use_numpy=True,
    )

    assert set(tensor.block(0).gradients_list()) == set(["cell", "positions"])

    tensor = equistore.remove_gradients(tensor, ["positions"])
    assert tensor.block(0).gradients_list() == ["cell"]
