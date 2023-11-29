import os

import metatensor


try:
    import torch  # noqa

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_detach():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )

    # just checking that everything runs fine with numpy
    tensor = metatensor.detach(tensor)

    if HAS_TORCH:
        tensor = metatensor.to(tensor, backend="torch")
        tensor = metatensor.requires_grad(tensor)

        for block in tensor:
            assert block.values.requires_grad

        output = metatensor.add(tensor, tensor)
        for block in output:
            assert block.values.requires_grad

        detached = metatensor.detach(output)
        for block in detached:
            assert not block.values.requires_grad
