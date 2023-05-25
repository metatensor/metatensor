from typing import Dict, List, Tuple, Union

import pytest
import torch

from equistore.torch import Labels, LabelsEntry, TensorBlock, TensorMap

from . import utils


@pytest.fixture
def tensor():
    return utils.tensor()


@pytest.fixture
def large_tensor():
    return utils.large_tensor()


def test_keys(tensor):
    assert tensor.keys.names == ["key_1", "key_2"]
    assert len(tensor.keys) == 4
    assert len(tensor) == 4

    expected = torch.tensor([[0, 0], [1, 0], [2, 2], [2, 3]])
    assert torch.all(tensor.keys.values == expected)


def test_print(tensor):
    expected = """TensorMap with 4 blocks
keys: key_1  key_2
        0      0
        1      0
        2      2
        2      3
"""
    assert expected == str(tensor)
    assert expected == tensor.print(6)


def test_print_large(large_tensor):
    expected = """TensorMap with 12 blocks
keys: key_1  key_2
        0      0
        1      0
          ...
        2      5
        3      5
"""
    assert expected == str(large_tensor)

    expected = """TensorMap with 12 blocks
keys: key_1  key_2
        0      0
        1      0
        2      2
          ...
        1      5
        2      5
        3      5
"""
    assert expected == large_tensor.print(6)


def test_labels_names(tensor):
    assert tensor.sample_names == ["s"]
    assert tensor.components_names == [["c"]]
    assert tensor.property_names == ["p"]


def test_block(tensor):
    # block by index
    block = tensor.block(2)
    assert torch.all(block.values == torch.full((4, 3, 1), 3.0))

    # block by index with __getitem__
    block = tensor[2]
    assert torch.all(block.values == torch.full((4, 3, 1), 3.0))

    # block by dict
    block = tensor.block(dict(key_1=1, key_2=0))
    assert torch.all(block.values == torch.full((3, 1, 3), 2.0))

    # block by Label entry
    block = tensor.block(tensor.keys[0])
    assert torch.all(block.values == torch.full((3, 1, 1), 1.0))

    # block by Label entry with __getitem__
    block = tensor[tensor.keys[0]]
    assert torch.all(block.values == torch.full((3, 1, 1), 1.0))

    # 0 blocks matching criteria
    msg = "could not find blocks matching the selection 'key_1=4, key_2=3'"
    with pytest.raises(ValueError, match=msg):
        tensor.block({"key_2": 3, "key_1": 4})

    # more than one block matching criteria
    msg = (
        "got more than one matching block for 'key_2=0', use the `blocks` "
        "function to select more than one block"
    )
    with pytest.raises(ValueError, match=msg):
        tensor.block({"key_2": 0})

    # Type errors
    msg = (
        "expected argument to be int, Dict\\[str, int\\], Labels, or "
        "LabelsEntry, got str"
    )
    with pytest.raises(ValueError, match=msg):
        tensor.block("key_2")

    msg = "expected argument to be Dict\\[str, int\\], got Dict\\[str, str\\]"
    with pytest.raises(ValueError, match=msg):
        tensor.block({"key_2": "0"})


def test_blocks(tensor):
    # block by index
    blocks = tensor.blocks(2)
    assert len(blocks) == 1
    assert torch.all(blocks[0].values == torch.full((4, 3, 1), 3.0))

    blocks = tensor.blocks([2, 3, 0])
    assert len(blocks) == 3
    assert torch.all(blocks[0].values == torch.full((4, 3, 1), 3.0))
    assert torch.all(blocks[1].values == torch.full((4, 3, 1), 4.0))
    assert torch.all(blocks[2].values == torch.full((3, 1, 1), 1.0))

    # block by kwargs
    blocks = tensor.blocks(dict(key_1=1, key_2=0))
    assert len(blocks) == 1
    assert torch.all(blocks[0].values == torch.full((3, 1, 3), 2.0))

    # more than one block
    blocks = tensor.blocks(dict(key_2=0))
    assert len(blocks) == 2

    assert torch.all(blocks[0].values == torch.full((3, 1, 1), 1.0))
    assert torch.all(blocks[1].values == torch.full((3, 1, 3), 2.0))

    # Type errors
    msg = (
        "expected argument to be None, int, List\\[int\\], Dict\\[str, int\\], "
        "Labels, or LabelsEntry, got str"
    )
    with pytest.raises(ValueError, match=msg):
        tensor.blocks("key_2")

    msg = "expected argument to be Dict\\[str, int\\], got Dict\\[str, str\\]"
    with pytest.raises(ValueError, match=msg):
        tensor.blocks({"key_2": "0"})


def test_iter(tensor):
    expected = [
        ((0, 0), torch.full((3, 1, 1), 1.0)),
        ((1, 0), torch.full((3, 1, 3), 2.0)),
        ((2, 2), torch.full((4, 3, 1), 3.0)),
        ((2, 3), torch.full((4, 3, 1), 4.0)),
    ]
    for i, (key, block) in enumerate(tensor.items()):
        expected_key, expected_values = expected[i]

        assert tuple(key) == expected_key
        assert torch.all(block.values == expected_values)

    # We can iterate over a TensorMap since it implements __len__ and __getitem__
    for i, block in enumerate(tensor):
        _, expected_values = expected[i]

        assert torch.all(block.values == expected_values)


def test_keys_to_properties(tensor):
    tensor = tensor.keys_to_properties("key_1")

    assert tensor.keys.names == ["key_2"]
    assert torch.all(tensor.keys.values == torch.tensor([(0,), (2,), (3,)]))

    # The new first block contains the old first two blocks merged
    block = tensor.block_by_id(0)
    assert tuple(block.samples.values[0]) == (0,)
    assert tuple(block.samples.values[1]) == (1,)
    assert tuple(block.samples.values[2]) == (2,)
    assert tuple(block.samples.values[3]) == (3,)
    assert tuple(block.samples.values[4]) == (4,)

    assert len(block.components), 1
    assert tuple(block.components[0].values[0]), (0,)

    assert block.properties.names == ["key_1", "p"]
    assert tuple(block.properties.values[0]) == (0, 0)
    assert tuple(block.properties.values[1]) == (1, 3)
    assert tuple(block.properties.values[2]) == (1, 4)
    assert tuple(block.properties.values[3]) == (1, 5)

    expected = torch.tensor(
        [
            [[1.0, 2.0, 2.0, 2.0]],
            [[0.0, 2.0, 2.0, 2.0]],
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 2.0, 2.0, 2.0]],
            [[1.0, 0.0, 0.0, 0.0]],
        ]
    )
    assert torch.all(block.values == expected)

    gradient = block.gradient("g")
    assert tuple(gradient.samples.values[0]) == (0, -2)
    assert tuple(gradient.samples.values[1]) == (0, 3)
    assert tuple(gradient.samples.values[2]) == (3, -2)
    assert tuple(gradient.samples.values[3]) == (4, 3)

    expected = torch.tensor(
        [
            [[11.0, 12.0, 12.0, 12.0]],
            [[0.0, 12.0, 12.0, 12.0]],
            [[0.0, 12.0, 12.0, 12.0]],
            [[11.0, 0.0, 0.0, 0.0]],
        ]
    )
    assert torch.all(gradient.values == expected)

    # The new second block contains the old third block
    block = tensor.block_by_id(1)
    assert block.properties.names == ["key_1", "p"]
    assert tuple(block.properties.values[0]) == (2, 0)

    assert torch.all(block.values == torch.full((4, 3, 1), 3.0))

    # The new third block contains the old fourth block
    block = tensor.block_by_id(2)
    assert block.properties.names == ["key_1", "p"]
    assert tuple(block.properties.values[0]) == (2, 0)

    assert torch.all(block.values == torch.full((4, 3, 1), 4.0))


def test_keys_to_samples(tensor):
    tensor = tensor.keys_to_samples("key_2", sort_samples=True)

    assert tensor.keys.names == ["key_1"]
    assert tuple(tensor.keys.values[0]) == (0,)
    assert tuple(tensor.keys.values[1]) == (1,)
    assert tuple(tensor.keys.values[2]) == (2,)

    # The first two blocks are not modified
    block = tensor.block_by_id(0)
    assert block.samples.names, ["s", "key_2"]
    assert tuple(block.samples.values[0]) == (0, 0)
    assert tuple(block.samples.values[1]) == (2, 0)
    assert tuple(block.samples.values[2]) == (4, 0)

    assert torch.all(block.values == torch.full((3, 1, 1), 1.0))

    block = tensor.block_by_id(1)
    assert block.samples.names == ["s", "key_2"]
    assert tuple(block.samples.values[0]) == (0, 0)
    assert tuple(block.samples.values[1]) == (1, 0)
    assert tuple(block.samples.values[2]) == (3, 0)

    assert torch.all(block.values == torch.full((3, 1, 3), 2.0))

    # The new third block contains the old third and fourth blocks merged
    block = tensor.block_by_id(2)

    assert block.samples.names == ["s", "key_2"]
    assert tuple(block.samples.values[0]) == (0, 2)
    assert tuple(block.samples.values[1]) == (0, 3)
    assert tuple(block.samples.values[2]) == (1, 3)
    assert tuple(block.samples.values[3]) == (2, 3)
    assert tuple(block.samples.values[4]) == (3, 2)
    assert tuple(block.samples.values[5]) == (5, 3)
    assert tuple(block.samples.values[6]) == (6, 2)
    assert tuple(block.samples.values[7]) == (8, 2)

    expected = torch.tensor(
        [
            [[3.0], [3.0], [3.0]],
            [[4.0], [4.0], [4.0]],
            [[4.0], [4.0], [4.0]],
            [[4.0], [4.0], [4.0]],
            [[3.0], [3.0], [3.0]],
            [[4.0], [4.0], [4.0]],
            [[3.0], [3.0], [3.0]],
            [[3.0], [3.0], [3.0]],
        ]
    )
    assert torch.all(block.values == expected)

    gradient = block.gradient("g")
    assert gradient.samples.names == ["sample", "g"]
    assert tuple(gradient.samples.values[0]) == (1, 1)
    assert tuple(gradient.samples.values[1]) == (4, -2)
    assert tuple(gradient.samples.values[2]) == (5, 3)

    expected = torch.tensor(
        [
            [[14.0], [14.0], [14.0]],
            [[13.0], [13.0], [13.0]],
            [[14.0], [14.0], [14.0]],
        ]
    )
    assert torch.all(gradient.values == expected)


def test_keys_to_samples_unsorted(tensor):
    tensor = tensor.keys_to_samples("key_2", sort_samples=False)

    block = tensor.block_by_id(2)
    assert block.samples.names == ["s", "key_2"]
    assert tuple(block.samples.values[0]) == (0, 2)
    assert tuple(block.samples.values[1]) == (3, 2)
    assert tuple(block.samples.values[2]) == (6, 2)
    assert tuple(block.samples.values[3]) == (8, 2)
    assert tuple(block.samples.values[4]) == (0, 3)
    assert tuple(block.samples.values[5]) == (1, 3)
    assert tuple(block.samples.values[6]) == (2, 3)
    assert tuple(block.samples.values[7]) == (5, 3)


def test_components_to_properties(tensor):
    tensor = tensor.components_to_properties("c")

    block = tensor.block_by_id(0)
    assert block.samples.names == ["s"]
    assert tuple(block.samples.values[0]) == (0,)
    assert tuple(block.samples.values[1]) == (2,)
    assert tuple(block.samples.values[2]) == (4,)

    assert block.components == []

    assert block.properties.names == ["c", "p"]
    assert tuple(block.properties.values[0]) == (0, 0)

    block = tensor.block_by_id(3)
    assert block.samples.names, ["s"]
    assert tuple(block.samples.values[0]) == (0,)
    assert tuple(block.samples.values[1]) == (1,)
    assert tuple(block.samples.values[2]) == (2,)
    assert tuple(block.samples.values[3]) == (5,)

    assert block.components == []

    assert block.properties.names == ["c", "p"]
    assert tuple(block.properties.values[0]) == (0, 0)
    assert tuple(block.properties.values[1]) == (1, 0)
    assert tuple(block.properties.values[2]) == (2, 0)


def test_empty_tensor():
    empty_tensor = TensorMap(keys=Labels.empty(["key"]), blocks=[])

    assert empty_tensor.keys.names == ["key"]

    assert empty_tensor.sample_names == []
    assert empty_tensor.components_names == []
    assert empty_tensor.property_names == []

    # TODO
    # assert empty_tensor.blocks() == []

    selection = Labels(names="key", values=torch.IntTensor([[3]]))
    assert empty_tensor.blocks_matching(selection) == []

    message = "invalid parameter: 'not_a_key' is not part of the keys for this tensor"
    with pytest.raises(RuntimeError, match=message):
        selection = Labels(names="not_a_key", values=torch.IntTensor([[3]]))
        empty_tensor.blocks_matching(selection)

    message = "block index out of bounds: we have 0 blocks but the index is 3"
    with pytest.raises(IndexError, match=message):
        empty_tensor.block_by_id(3)

    message = "invalid parameter: there are no keys to move in an empty TensorMap"
    with pytest.raises(RuntimeError, match=message):
        empty_tensor.keys_to_samples("key")

    with pytest.raises(RuntimeError, match=message):
        empty_tensor.keys_to_properties("key")


# define a wrapper class to make sure the types TorchScript uses for of all
# C-defined functions matches what we expect
class TensorMapWrap:
    def __init__(self, keys: Labels, blocks: List[TensorBlock]):
        self._c = TensorMap(keys=keys, blocks=blocks)

    def __len__(self) -> int:
        return self._c.__len__()

    def __str__(self) -> str:
        return self._c.__str__()

    def __repr__(self) -> str:
        return self._c.__repr__()

    def copy(self) -> TensorMap:
        return self._c.copy()

    def items(self) -> List[Tuple[LabelsEntry, TensorBlock]]:
        return self._c.items()

    def keys(self) -> Labels:
        return self._c.keys

    def blocks_matching(self, selection: Labels) -> List[int]:
        return self._c.blocks_matching(selection=selection)

    def block_by_id(self, index: int) -> TensorBlock:
        return self._c.block_by_id(index=index)

    def blocks_by_id(self, indices: List[int]) -> List[TensorBlock]:
        return self._c.blocks_by_id(indices=indices)

    def block(
        self, selection: Union[int, Dict[str, int], Labels, LabelsEntry]
    ) -> TensorBlock:
        return self._c.block(selection=selection)

    def blocks(
        self,
        selection: Union[None, List[int], Dict[str, int], Labels, LabelsEntry] = None,
    ) -> List[TensorBlock]:
        return self._c.blocks(selection=selection)

    def keys_to_samples(
        self,
        keys_to_move: Union[str, List[str], Labels],
        sort_samples: bool,
    ) -> TensorMap:
        return self._c.keys_to_samples(
            keys_to_move=keys_to_move,
            sort_samples=sort_samples,
        )

    def keys_to_properties(
        self,
        keys_to_move: Union[str, List[str], Labels],
        sort_samples: bool,
    ) -> TensorMap:
        return self._c.keys_to_properties(
            keys_to_move=keys_to_move,
            sort_samples=sort_samples,
        )

    def components_to_properties(
        self,
        dimensions: Union[str, List[str]],
    ) -> TensorMap:
        return self._c.components_to_properties(dimensions=dimensions)

    def sample_names(self) -> List[str]:
        return self._c.sample_names

    def components_names(self) -> List[List[str]]:
        return self._c.components_names

    def property_names(self) -> List[str]:
        return self._c.property_names

    def print_(self, max_keys: int) -> str:
        return self._c.print(max_keys=max_keys)


def test_script():
    class TestModule(torch.nn.Module):
        def forward(self, x: TensorMapWrap) -> TensorMapWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)
