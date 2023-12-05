from typing import List, Tuple

import pytest
import torch
from packaging import version
from torch import Tensor

from metatensor.torch import Labels, TensorBlock


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


def test_repr():
    block = TensorBlock(
        values=torch.full((3, 2), 11),
        samples=Labels(names=["s"], values=torch.IntTensor([[0], [2], [1]])),
        components=[],
        properties=Labels(names=["p"], values=torch.IntTensor([[1], [0]])),
    )

    expected = """TensorBlock
    samples (3): ['s']
    components (): []
    properties (2): ['p']
    gradients: None
"""

    assert str(block) == expected

    block = TensorBlock(
        values=torch.full((3, 3, 1, 2, 5), 11),
        samples=Labels(
            names=["s_1", "s_2"], values=torch.IntTensor([[0, 0], [1, 1], [2, 2]])
        ),
        components=[
            Labels(names=["c_1"], values=torch.IntTensor([[0], [1], [2]])),
            Labels(names=["c_2"], values=torch.IntTensor([[0]])),
            Labels(names=["c_3"], values=torch.IntTensor([[0], [1]])),
        ],
        properties=Labels(
            names=["p_1", "p_2"],
            values=torch.IntTensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
        ),
    )

    block.add_gradient(
        "g",
        TensorBlock(
            values=torch.full((3, 3, 1, 2, 5), 11),
            samples=Labels(
                names=["sample", "g"], values=torch.IntTensor([[0, 0], [1, 1], [2, 2]])
            ),
            components=[
                Labels(names=["c_1"], values=torch.IntTensor([[0], [1], [2]])),
                Labels(names=["c_2"], values=torch.IntTensor([[0]])),
                Labels(names=["c_3"], values=torch.IntTensor([[0], [1]])),
            ],
            properties=Labels(
                names=["p_1", "p_2"],
                values=torch.IntTensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
            ),
        ),
    )

    expected = """TensorBlock
    samples (3): ['s_1', 's_2']
    components (3, 1, 2): ['c_1', 'c_2', 'c_3']
    properties (5): ['p_1', 'p_2']
    gradients: ['g']
"""
    assert str(block) == expected

    if version.parse(torch.__version__) >= version.parse("2.1"):
        # custom __repr__ definitions are only available since torch 2.1
        assert repr(block) == expected

    expected = """Gradient TensorBlock ('g')
    samples (3): ['sample', 'g']
    components (3, 1, 2): ['c_1', 'c_2', 'c_3']
    properties (5): ['p_1', 'p_2']
    gradients: None
"""
    assert str(block.gradient("g")) == expected

    if version.parse(torch.__version__) >= version.parse("2.1"):
        # custom __repr__ definitions are only available since torch 2.1
        assert repr(block.gradient("g")) == expected


def test_copy():
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
    assert isinstance(gradients, list)
    assert len(gradients) == 1
    assert gradients[0][0] == "g"


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

    def __str__(self) -> str:
        return self._c.__str__()

    def __repr__(self) -> str:
        return self._c.__repr__()

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

    def gradients(self) -> List[Tuple[str, TensorBlock]]:
        return self._c.gradients()


def test_script():
    class TestModule(torch.nn.Module):
        def forward(self, x: TensorBlockWrap) -> TensorBlockWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)


def test_different_device():
    with pytest.raises(
        ValueError,
        match="cannot create TensorBlock: values and samples must "
        "be on the same device, got meta and cpu",
    ):
        TensorBlock(
            values=torch.tensor([[[3.0, 4.0]]], device="meta"),
            samples=Labels.range("samples", 1),
            components=[Labels.range("component", 1)],
            properties=Labels.range("properties", 2),
        )


def test_different_dtype_gradient():
    message = (
        "the gradient and the original block must have the same dtype, "
        "got torch.float16 and torch.float32"
    )
    with pytest.raises(TypeError, match=message):
        block = TensorBlock(
            values=torch.tensor([[[3.0, 4.0]]]),
            samples=Labels.range("samples", 1),
            components=[Labels.range("component", 1)],
            properties=Labels.range("properties", 2),
        )
        block.add_gradient(
            "gradient",
            TensorBlock(
                values=torch.tensor([[[3.0, 4.0]]], dtype=torch.float16),
                samples=Labels.range("samples", 1),
                components=[Labels.range("component", 1)],
                properties=Labels.range("properties", 2),
            ),
        )
