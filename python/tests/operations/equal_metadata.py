import os

import numpy as np
import pytest

import equistore
from equistore import Labels, TensorBlock, TensorMap, equal_metadata


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def test_tensor_map_1() -> TensorMap:
    return equistore.io.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        # the npz is using DEFLATE compression, equistore only supports STORED
        use_numpy=True,
    )


@pytest.fixture
def test_tensor_map_2() -> TensorMap:
    return equistore.io.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, equistore only supports STORED
        use_numpy=True,
    )


@pytest.fixture
def tensor_map() -> TensorMap:
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    block_1 = TensorBlock(
        values=np.full((3, 1, 1), 1.0),
        samples=Labels(["samples"], np.array([[0], [2], [4]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_1.add_gradient(
        "parameter",
        samples=Labels(
            ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
        ),
        data=np.full((2, 1, 1), 11.0),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 2.0),
        samples=Labels(["samples"], np.array([[0], [1], [3]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[3], [4], [5]], dtype=np.int32)),
    )
    block_2.add_gradient(
        "parameter",
        data=np.full((3, 1, 3), 12.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, -2], [0, 3], [2, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_3 = TensorBlock(
        values=np.full((4, 3, 1), 3.0),
        samples=Labels(["samples"], np.array([[0], [3], [6], [8]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_3.add_gradient(
        "parameter",
        data=np.full((1, 3, 1), 13.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[1, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
    )

    block_4 = TensorBlock(
        values=np.full((4, 3, 1), 4.0),
        samples=Labels(["samples"], np.array([[0], [1], [2], [5]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_4.add_gradient(
        "parameter",
        data=np.full((2, 3, 1), 14.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, 1], [3, 3]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
    )

    # TODO: different number of components?

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array([[0, 0], [1, 0], [2, 2], [2, 3]], dtype=np.int32),
    )

    return TensorMap(keys, [block_1, block_2, block_3, block_4])


class TestEqualMetaData:
    def test_self(self, test_tensor_map_1):
        # check if the metadata of the same tensor are equal
        assert equal_metadata(test_tensor_map_1, test_tensor_map_1)

    def test_two_tesnors(self, test_tensor_map_1, test_tensor_map_2):
        # check if the metadata of two tensor maps are equal
        assert not equal_metadata(test_tensor_map_1, test_tensor_map_2)

    def test_after_drop(self, test_tensor_map_1):
        # check if dropping an existing block changes the metadata
        new_key = Labels(
            test_tensor_map_1.keys.names, np.array([tuple(test_tensor_map_1.keys[0])])
        )
        new_tesnor = equistore.drop_blocks(test_tensor_map_1, new_key)
        assert not equal_metadata(test_tensor_map_1, new_tesnor)

    def test_single_nonexisting_meta(self, test_tensor_map_1, test_tensor_map_2):
        # check behavior if non existing metadata is provided
        # wrong metadata key alone
        wrong_meta = "species"
        error_message = f"Invalid metadata to check: {wrong_meta}"
        with pytest.raises(ValueError, match=error_message):
            equal_metadata(
                tensor_1=test_tensor_map_1,
                tensor_2=test_tensor_map_1,
                check=[wrong_meta],
            )
        with pytest.raises(ValueError, match=error_message):
            equal_metadata(
                tensor_1=test_tensor_map_1,
                tensor_2=test_tensor_map_2,
                check=[wrong_meta],
            )
        # wrong metadata key with another correct one
        correct_meta = "properties"
        with pytest.raises(ValueError, match=error_message):
            equal_metadata(
                tensor_1=test_tensor_map_1,
                tensor_2=test_tensor_map_1,
                check=[correct_meta, wrong_meta],
            )
        with pytest.raises(ValueError, match=error_message):
            equal_metadata(
                tensor_1=test_tensor_map_1,
                tensor_2=test_tensor_map_2,
                check=[correct_meta, wrong_meta],
            )

    def test_changing_tensor_key_order(self, test_tensor_map_1):
        # check changing the key order
        keys = test_tensor_map_1.keys
        new_keys = keys[::-1]
        new_blocks = [test_tensor_map_1[key].copy() for key in new_keys]
        new_tensor = TensorMap(keys=new_keys, blocks=new_blocks)
        assert equal_metadata(test_tensor_map_1, new_tensor)

    def test_changing_samples_key_order(self, test_tensor_map_1):
        # changing the order of the values of the samples should yield False
        new_blocks = []
        for key in test_tensor_map_1.keys:
            block = test_tensor_map_1[key].copy()
            samples = block.samples[::-1]
            new_block = TensorBlock(
                values=block.values,
                samples=samples,
                properties=block.properties,
                components=block.components,
            )
            for param, obj in block.gradients():
                new_block.add_gradient(
                    parameter=param,
                    samples=obj.samples,
                    components=obj.components,
                    data=obj.data,
                )
            new_blocks.append(new_block)

        new_tensor = TensorMap(keys=test_tensor_map_1.keys, blocks=new_blocks)
        assert not equal_metadata(test_tensor_map_1, new_tensor)

    def test_changing_properties_key_order(self, test_tensor_map_1):
        # changing the order of the values of the properties should yield False
        new_blocks = []
        for key in test_tensor_map_1.keys:
            block = test_tensor_map_1[key].copy()
            properties = block.properties[::-1]
            new_block = TensorBlock(
                values=block.values,
                samples=block.samples,
                properties=properties,
                components=block.components,
            )
            for param, obj in block.gradients():
                new_block.add_gradient(
                    parameter=param,
                    samples=obj.samples,
                    components=obj.components,
                    data=obj.data,
                )
            new_blocks.append(new_block)

        new_tensor = TensorMap(keys=test_tensor_map_1.keys, blocks=new_blocks)
        assert not equal_metadata(test_tensor_map_1, new_tensor)

    def test_add_components_key_order(self, tensor_map):
        # changing the order of the values of the components should yield False
        new_blocks = []
        for key in tensor_map.keys:
            block = tensor_map[key].copy()
            components = [comp[::-1] for comp in block.components]
            new_block = TensorBlock(
                values=block.values,
                samples=block.samples,
                properties=block.properties,
                components=components,
            )
            for param, obj in block.gradients():
                new_block.add_gradient(
                    parameter=param,
                    samples=obj.samples,
                    components=components,
                    data=obj.data,
                )
            new_blocks.append(new_block)

        new_tensor = TensorMap(keys=tensor_map.keys, blocks=new_blocks)
        assert not equal_metadata(tensor_map, new_tensor)

    def test_remove_last_sample(self, tensor_map):
        # removing the last sample should yield False
        new_blocks = []
        for key in tensor_map.keys:
            block = tensor_map[key].copy()
            new_block = TensorBlock(
                values=block.values[:-1],
                samples=block.samples[:-1],
                properties=block.properties,
                components=block.components,
            )
            for param, obj in block.gradients():
                new_block.add_gradient(
                    parameter=param,
                    samples=obj.samples[:-1],
                    components=obj.components,
                    data=obj.data[:-1],
                )
            new_blocks.append(new_block)

        new_tensor = TensorMap(keys=tensor_map.keys, blocks=new_blocks)
        assert not equal_metadata(tensor_map, new_tensor)

    def test_remove_last_property(self, tensor_map):
        # removing the last property should yield False
        new_blocks = []
        for key in tensor_map.keys:
            block = tensor_map[key].copy()
            new_block = TensorBlock(
                values=block.values[..., :-1],
                samples=block.samples,
                properties=block.properties[..., :-1],
                components=block.components,
            )
            for param, obj in block.gradients():
                new_block.add_gradient(
                    parameter=param,
                    samples=obj.samples,
                    components=obj.components,
                    data=obj.data[..., :-1],
                )
            new_blocks.append(new_block)

        new_tensor = TensorMap(keys=tensor_map.keys, blocks=new_blocks)
        assert not equal_metadata(tensor_map, new_tensor)

    def test_remove_last_component(self, tensor_map):
        # removing the last component should yield False
        new_blocks = []
        for key in tensor_map.keys:
            block = tensor_map[key].copy()
            components = [comp[:-1] for comp in block.components]
            new_block = TensorBlock(
                values=block.values[:, :-1],
                samples=block.samples,
                properties=block.properties,
                components=components,
            )
            for param, obj in block.gradients():
                new_block.add_gradient(
                    parameter=param,
                    samples=obj.samples,
                    components=components,
                    data=obj.data[:, :-1],
                )
            new_blocks.append(new_block)

        new_tensor = TensorMap(keys=tensor_map.keys, blocks=new_blocks)
        assert not equal_metadata(tensor_map, new_tensor)
