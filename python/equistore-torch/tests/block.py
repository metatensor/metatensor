from typing import Dict, List

import pytest
import torch
from torch import Tensor

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
    assert block.samples.names == ["s"]
    assert block.components == []
    assert block.properties.names == ["p"]

    # positional arguments
    block = TensorBlock(
        torch.full((3, 2), 33),
        Labels(names=["s"], values=torch.IntTensor([[0], [2], [1]])),
        [],
        Labels(names=["p"], values=torch.IntTensor([[1], [0]])),
    )

    assert torch.all(block.values == torch.full((3, 2), 33))
    assert block.samples.names == ["s"]
    assert block.components == []
    assert block.properties.names == ["p"]


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
    assert gradient.samples.names == ["sample", "g"]

    message = "can not find gradients with respect to 'not-there' in this block"
    with pytest.raises(RuntimeError, match=message):
        gradient = block.gradient("not-there")

    gradients = block.gradients()
    assert isinstance(gradients, dict)
    assert list(gradients.keys()) == ["g"]


# define a wrapper class to make sure the types TorchScript uses for of all
# C-defined functions matches what we expect
class TensorBlockWrap:
    def __init__(
        self,
        values: Tensor,
        samples: Labels,
        components: List[Labels],
        properties: Labels,
    ):
        self._c = TensorBlock(
            values=values,
            samples=samples,
            components=components,
            properties=properties,
        )

    def copy(self) -> TensorBlock:
        return self._c.copy()

    def values(self) -> Tensor:
        return self._c.values

    def samples(self) -> Labels:
        return self._c.samples

    def components(self) -> List[Labels]:
        return self._c.components

    def properties(self) -> Labels:
        return self._c.properties

    def add_gradient(self, parameter: str, gradient: TensorBlock):
        return self._c.add_gradient(parameter=parameter, gradient=gradient)

    def gradients_list(self) -> List[str]:
        return self._c.gradients_list()

    def has_gradient(self, parameter: str) -> bool:
        return self._c.has_gradient(parameter=parameter)

    def gradient(self, parameter: str) -> TensorBlock:
        return self._c.gradient(parameter=parameter)

    def gradients(self) -> Dict[str, TensorBlock]:
        return self._c.gradients()


def test_script():
    class TestModule(torch.nn.Module):
        def forward(self, x: TensorBlockWrap) -> TensorBlockWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)
