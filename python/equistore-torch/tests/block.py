import pytest
import torch

from equistore.torch import Labels, TensorBlock


def test_constructor():
    # keyword arguments
    block = TensorBlock(
        values=torch.full((3, 2), 11),
        samples=Labels(names=["s"], values=torch.IntTensor([[0], [2], [1]])),
        components=[],
        properties=Labels(names=["p"], values=torch.IntTensor([[1], [0]])),
    )

    assert torch.all(block.values == torch.full((3, 2), 11))
    assert block.samples.names == ("s",)
    assert block.components == []
    assert block.properties.names == ("p",)

    # positional arguments
    block = TensorBlock(
        torch.full((3, 2), 33),
        Labels(names=["s"], values=torch.IntTensor([[0], [2], [1]])),
        [],
        Labels(names=["p"], values=torch.IntTensor([[1], [0]])),
    )

    assert torch.all(block.values == torch.full((3, 2), 33))
    assert block.samples.names == ("s",)
    assert block.components == []
    assert block.properties.names == ("p",)


def test_clone():
    values = torch.full((3, 2), 11)
    block = TensorBlock(
        values=values,
        samples=Labels(names=["s"], values=torch.IntTensor([[0], [2], [1]])),
        components=[],
        properties=Labels(names=["p"], values=torch.IntTensor([[1], [0]])),
    )

    assert values.data_ptr() == block.values.data_ptr()

    clone = block.copy()
    del block

    assert values.data_ptr() != clone.values.data_ptr()


def test_gradients():
    block = TensorBlock(
        values=torch.full((3, 2), 1),
        samples=Labels(names=["s"], values=torch.IntTensor([[0], [2], [1]])),
        components=[],
        properties=Labels(names=["p"], values=torch.IntTensor([[1], [0]])),
    )

    assert block.gradients_list() == []

    block.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.full((1, 3, 2), 11),
            samples=Labels(
                names=("sample", "g"),
                values=torch.IntTensor([[0, 1]]),
            ),
            components=[Labels(names=("c"), values=torch.IntTensor([[0], [1], [2]]))],
            properties=block.properties,
        ),
    )

    assert block.gradients_list() == ["g"]
    assert block.has_gradient("g") is True
    assert block.has_gradient(parameter="not-there") is False

    gradient = block.gradient("g")
    assert gradient.values.shape == (1, 3, 2)

    gradient = block.gradient(parameter="g")
    assert gradient.samples.names == ("sample", "g")

    message = "can not find gradients with respect to 'not-there' in this block"
    with pytest.raises(RuntimeError, match=message):
        gradient = block.gradient("not-there")

    gradients = block.gradients()
    assert isinstance(gradients, dict)
    assert list(gradients.keys()) == ["g"]
