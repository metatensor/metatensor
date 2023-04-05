import numpy as np

from equistore import Labels, TensorBlock


def test_nested():
    """
    Test that nested gradients are correctly returned when accessed via their
    relative syntax
    """
    grad = np.random.rand(1, 1)
    grad_grad = np.random.rand(1, 1)
    block = TensorBlock(
        values=np.array([[0.0]]),
        samples=Labels.single(),
        components=[],
        properties=Labels.single(),
    )
    block.add_gradient(
        parameter="gradient",
        values=grad,
        samples=Labels(["sample"], np.array([[0]])),
        components=[],
    )
    block.gradient("gradient").add_gradient(
        parameter="gradient_of_gradient",
        values=grad_grad,
        samples=Labels(["sample"], np.array([[0]])),
        components=[],
    )
    returned_grad_grad = block.gradient("gradient/gradient_of_gradient").values
    assert np.allclose(grad_grad, returned_grad_grad)
