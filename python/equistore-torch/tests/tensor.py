import pytest
import torch

from equistore.torch import Labels, TensorMap

from . import utils


@pytest.fixture
def tensor():
    return utils.tensor()


def test_keys(tensor):
    assert tensor.keys.names == ("key_1", "key_2")
    assert len(tensor.keys) == 4
    assert len(tensor) == 4

    expected = torch.tensor([[0, 0], [1, 0], [2, 2], [2, 3]])
    assert torch.all(tensor.keys.values == expected)


@pytest.mark.skip("not implemented")
def test_print(tensor):
    """
    Test routine for the print function of the TensorBlock.
    It compare the results with those in a file.
    """
    repr = tensor.__repr__()
    expected = """TensorMap with 4 blocks
keys: ['key_1' 'key_2']
          0       0
          1       0
          2       2
          2       3"""
    assert expected == repr


@pytest.mark.skip("not implemented")
def test_print_large(large_tensor):
    _print = large_tensor.__repr__()
    expected = """TensorMap with 12 blocks
keys: ['key_1' 'key_2']
          0       0
          1       0
          2       2
       ...
          1       5
          2       5
          3       5"""
    assert expected == _print


def test_labels_names(tensor):
    assert tensor.sample_names == ("s",)
    assert tensor.components_names == [("c",)]
    assert tensor.property_names == ("p",)


@pytest.mark.skip("not implemented")
def test_block(tensor):
    # block by index
    block = tensor.block(2)
    assert torch.all(block.values == torch.full((4, 3, 1), 3.0))

    # block by index with __getitem__
    block = tensor[2]
    assert torch.all(block.values == torch.full((4, 3, 1), 3.0))

    # block by kwargs
    block = tensor.block(key_1=1, key_2=0)
    assert torch.all(block.values == torch.full((3, 1, 3), 2.0))

    # block by Label entry
    block = tensor.block(tensor.keys[0])
    assert torch.all(block.values == torch.full((3, 1, 1), 1.0))

    # block by Label entry with __getitem__
    block = tensor[tensor.keys[0]]
    assert torch.all(block.values == torch.full((3, 1, 1), 1.0))

    # More arguments than needed: two integers
    # by index
    msg = "only one non-keyword argument is supported, 2 are given"
    with pytest.raises(ValueError, match=msg):
        tensor.block(3, 4)

    # 4 input with the first as integer by __getitem__
    msg = "only one non-keyword argument is supported, 4 are given"
    with pytest.raises(ValueError, match=msg):
        tensor[3, 4, 7.0, "r"]

    # More arguments than needed: 3 Labels
    msg = "only one non-keyword argument is supported, 3 are given"
    with pytest.raises(ValueError, match=msg):
        tensor.block(tensor.keys[0], tensor.keys[1], tensor.keys[3])

    # by __getitem__
    msg = "only one non-keyword argument is supported, 2 are given"
    with pytest.raises(ValueError, match=msg):
        tensor[tensor.keys[1], 4]

    # 0 blocks matching criteria
    msg = "Couldn't find any block matching the selection 'key_1 = 3'"
    with pytest.raises(ValueError, match=msg):
        tensor.block(key_1=3)

    # more than one block matching criteria
    msg = (
        "more than one block matched 'key_2 = 0', use `TensorMap.blocks` "
        "if you want to get all of them"
    )
    with pytest.raises(ValueError, match=msg):
        tensor.block(key_2=0)


@pytest.mark.skip("not implemented")
def test_blocks(tensor):
    # block by index
    blocks = tensor.blocks(2)
    assert len(blocks) == 1
    assert torch.all(blocks[0].values == torch.full((4, 3, 1), 3.0))

    # block by kwargs
    blocks = tensor.blocks(key_1=1, key_2=0)
    assert len(blocks) == 1
    assert torch.all(blocks[0].values == torch.full((3, 1, 3), 2.0))

    # more than one block
    blocks = tensor.blocks(key_2=0)
    assert len(blocks) == 2

    assert torch.all(blocks[0].values == torch.full((3, 1, 1), 1.0))
    assert torch.all(blocks[1].values == torch.full((3, 1, 3), 2.0))


@pytest.mark.skip("not implemented")
def test_iter(tensor):
    expected = [
        ((0, 0), torch.full((3, 1, 1), 1.0)),
        ((1, 0), torch.full((3, 1, 3), 2.0)),
        ((2, 2), torch.full((4, 3, 1), 3.0)),
        ((2, 3), torch.full((4, 3, 1), 4.0)),
    ]
    for i, (sparse, block) in enumerate(tensor):
        expected_sparse, expected_values = expected[i]

        assert tuple(sparse) == expected_sparse
        assert torch.all(block.values == expected_values)


def test_keys_to_properties(tensor):
    tensor = tensor.keys_to_properties("key_1")

    assert tensor.keys.names == ("key_2",)
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

    assert block.properties.names == ("key_1", "p")
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
    assert block.properties.names == ("key_1", "p")
    assert tuple(block.properties.values[0]) == (2, 0)

    assert torch.all(block.values == torch.full((4, 3, 1), 3.0))

    # The new third block contains the old fourth block
    block = tensor.block_by_id(2)
    assert block.properties.names == ("key_1", "p")
    assert tuple(block.properties.values[0]) == (2, 0)

    assert torch.all(block.values == torch.full((4, 3, 1), 4.0))


def test_keys_to_samples(tensor):
    tensor = tensor.keys_to_samples("key_2", sort_samples=True)

    assert tensor.keys.names == ("key_1",)
    assert tuple(tensor.keys.values[0]) == (0,)
    assert tuple(tensor.keys.values[1]) == (1,)
    assert tuple(tensor.keys.values[2]) == (2,)

    # The first two blocks are not modified
    block = tensor.block_by_id(0)
    assert block.samples.names, ("s", "key_2")
    assert tuple(block.samples.values[0]) == (0, 0)
    assert tuple(block.samples.values[1]) == (2, 0)
    assert tuple(block.samples.values[2]) == (4, 0)

    assert torch.all(block.values == torch.full((3, 1, 1), 1.0))

    block = tensor.block_by_id(1)
    assert block.samples.names == ("s", "key_2")
    assert tuple(block.samples.values[0]) == (0, 0)
    assert tuple(block.samples.values[1]) == (1, 0)
    assert tuple(block.samples.values[2]) == (3, 0)

    assert torch.all(block.values == torch.full((3, 1, 3), 2.0))

    # The new third block contains the old third and fourth blocks merged
    block = tensor.block_by_id(2)

    assert block.samples.names == ("s", "key_2")
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
    assert gradient.samples.names == ("sample", "g")
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
    assert block.samples.names, ("s", "key_2")
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
    assert block.samples.names == ("s",)
    assert tuple(block.samples.values[0]) == (0,)
    assert tuple(block.samples.values[1]) == (2,)
    assert tuple(block.samples.values[2]) == (4,)

    assert block.components == []

    assert block.properties.names == ("c", "p")
    assert tuple(block.properties.values[0]) == (0, 0)

    block = tensor.block_by_id(3)
    assert block.samples.names, ("s",)
    assert tuple(block.samples.values[0]) == (0,)
    assert tuple(block.samples.values[1]) == (1,)
    assert tuple(block.samples.values[2]) == (2,)
    assert tuple(block.samples.values[3]) == (5,)

    assert block.components == []

    assert block.properties.names == ("c", "p")
    assert tuple(block.properties.values[0]) == (0, 0)
    assert tuple(block.properties.values[1]) == (1, 0)
    assert tuple(block.properties.values[2]) == (2, 0)


@pytest.mark.skip("Labels.empty not implemented")
def test_empty_tensor():
    empty_tensor = TensorMap(keys=Labels.empty(["key"]), blocks=[])

    assert empty_tensor.keys.names == ("key",)

    assert empty_tensor.sample_names == tuple()
    assert empty_tensor.components_names == []
    assert empty_tensor.property_names == tuple()

    assert empty_tensor.blocks() == []

    assert empty_tensor.blocks_matching(key=3) == []

    # message = "invalid parameter: 'not_a_key' is not part of the keys for this tensor"
    # with pytest.raises(EquistoreError, match=message):
    #     empty_tensor.blocks_matching(not_a_key=3)

    # message = (
    #     "invalid parameter: block index out of bounds: we have "
    #     "0 blocks but the index is 3"
    # )
    # with pytest.raises(EquistoreError, match=message):
    #     empty_tensor.block(3)

    # message = "invalid parameter: there are no keys to move in an empty TensorMap"
    # with pytest.raises(EquistoreError, match=message):
    #     empty_tensor.keys_to_samples("key")

    # with pytest.raises(EquistoreError, match=message):
    #     empty_tensor.keys_to_properties("key")
