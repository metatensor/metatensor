import os

import metatensor as mts


try:
    import torch  # noqa

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_detach():
    tensor = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))

    # just checking that everything runs fine with numpy
    tensor = mts.detach(tensor)

    if HAS_TORCH:
        tensor = tensor.to(arrays="torch")
        tensor = mts.requires_grad(tensor)

        for block in tensor:
            assert block.values.requires_grad

        output = mts.add(tensor, tensor)
        for block in output:
            assert block.values.requires_grad

        detached = mts.detach(output)
        for block in detached:
            assert not block.values.requires_grad
