import os

import numpy as np

import metatensor as mts
from metatensor import Labels, TensorBlock


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_sum_samples_block():
    tensor_se = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))
    tensor_ps = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))

    block_ps = tensor_ps[0]
    # check both passing a list and a single string for sample_names
    reduce_block_ps = mts.sum_over_samples_block(block_ps, sample_names=["atom"])

    assert np.all(np.sum(block_ps.values[:4], axis=0) == reduce_block_ps.values[0])

    assert np.allclose(
        np.sum(block_ps.values[4:10], axis=0),
        reduce_block_ps.values[1],
        rtol=1e-13,
    )

    assert np.all(np.sum(block_ps.values[22:26], axis=0) == reduce_block_ps.values[5])

    assert np.all(np.sum(block_ps.values[38:46], axis=0) == reduce_block_ps.values[8])

    assert np.allclose(
        np.sum(block_ps.values[46:], axis=0),
        reduce_block_ps.values[9],
        rtol=1e-13,
    )

    # Test the gradients
    gradient = block_ps.gradient("positions").values

    assert np.allclose(
        np.sum(gradient[[0, 4, 8, 12]], axis=0),
        reduce_block_ps.gradient("positions").values[0],
        rtol=1e-13,
    )
    assert np.allclose(
        np.sum(gradient[[2, 6, 10, 14]], axis=0),
        reduce_block_ps.gradient("positions").values[2],
    )

    assert np.all(
        np.sum(gradient[[3, 7, 11, 15]], axis=0)
        == reduce_block_ps.gradient("positions").values[3]
    )

    assert np.allclose(
        np.sum(gradient[[96, 99, 102]], axis=0),
        reduce_block_ps.gradient("positions").values[40],
        rtol=1e-13,
    )

    assert np.all(
        np.sum(gradient[-1], axis=0) == reduce_block_ps.gradient("positions").values[-1]
    )

    # The TensorBlock with key=(8,8,8) has nothing to be averaged over

    for _ii, bl2 in enumerate([tensor_se[0], tensor_se[1], tensor_se[2], tensor_se[3]]):
        reduced_block = mts.sum_over_samples_block(bl2, sample_names="atom")
        assert np.all(np.sum(bl2.values[:4], axis=0) == reduced_block.values[0])
        assert np.allclose(
            np.sum(bl2.values[26:32], axis=0),
            reduced_block.values[6],
            rtol=1e-13,
        )

        assert np.allclose(
            np.sum(bl2.values[32:38], axis=0),
            reduced_block.values[7],
            rtol=1e-13,
        )
        assert np.allclose(
            np.sum(bl2.values[46:], axis=0),
            reduced_block.values[9],
            rtol=1e-13,
        )


def test_sum_properties_block():
    tensor_se = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))
    tensor_ps = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))
    bl1 = tensor_ps[0]

    # check both passing a list and a single string for property names
    reduce_block_ps = mts.sum_over_properties_block(bl1, property_names="l")

    assert np.all(np.sum(bl1.values[:, ::16], axis=-1) == reduce_block_ps.values[:, 0])
    assert np.all(np.sum(bl1.values[:, 2::16], axis=-1) == reduce_block_ps.values[:, 2])
    assert np.all(np.sum(bl1.values[:, 5::16], axis=-1) == reduce_block_ps.values[:, 5])
    assert np.all(np.sum(bl1.values[:, 8::16], axis=-1) == reduce_block_ps.values[:, 8])
    assert np.all(np.sum(bl1.values[:, 9::16], axis=-1) == reduce_block_ps.values[:, 9])

    # Test the gradients
    gr1 = tensor_ps[0].gradient("positions").values

    assert np.all(
        np.sum(gr1[..., ::16], axis=-1)
        == reduce_block_ps.gradient("positions").values[..., 0]
    )
    assert np.all(
        np.sum(gr1[..., 2::16], axis=-1)
        == reduce_block_ps.gradient("positions").values[..., 2]
    )

    assert np.all(
        np.sum(gr1[..., 3::16], axis=-1)
        == reduce_block_ps.gradient("positions").values[..., 3]
    )

    assert np.all(
        np.sum(gr1[..., 14::16], axis=-1)
        == reduce_block_ps.gradient("positions").values[..., 14]
    )

    for _, bl2 in enumerate([tensor_se[0], tensor_se[1], tensor_se[2], tensor_se[3]]):
        reduce_block_se = mts.sum_over_properties_block(bl2, property_names=["n"])
        assert np.all(
            np.sum(bl2.values, axis=-1, keepdims=True)[..., 0]
            == reduce_block_se.values[..., 0]
        )


def test_reduction_block_multi_samples():
    """Test sum over samples for multiple dimensions simultaneously"""
    block_1 = TensorBlock(
        values=np.array(
            [
                [1, 2, 4],
                [3, 5, 6],
                [-1.3, 26.7, 4.54],
                [3.5, 5.3, 6.87],
                [6.1, 35.2, 44.5],
                [7.3, -7.65, 6.45],
                [11, 276.0, 4.09],
                [33, 55.5, -5.6],
            ]
        ),
        samples=Labels(
            ["s1", "s2", "s3"],
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 1, 1],
                    [0, 1, 0],
                    [2, 1, 1],
                    [1, 1, 1],
                    [1, 0, 0],
                ],
            ),
        ),
        components=[],
        properties=Labels(["p"], np.array([[0], [1], [5]])),
    )

    reduce_block_12 = mts.sum_over_samples_block(block_1, sample_names=["s3"])
    reduce_block_23 = mts.sum_over_samples_block(block_1, sample_names="s1")
    reduce_block_2 = mts.sum_over_samples_block(block_1, sample_names=["s1", "s3"])

    assert np.allclose(
        np.sum(block_1.values[:3], axis=0),
        reduce_block_12.values[0],
        rtol=1e-13,
    )

    assert np.all(np.sum(block_1.values[3:5], axis=0) == reduce_block_12.values[1])
    assert np.all(block_1.values[5] == reduce_block_12.values[4])
    assert np.all(block_1.values[6] == reduce_block_12.values[3])
    assert np.all(block_1.values[7] == reduce_block_12.values[2])

    assert np.all(np.sum(block_1.values[[0, 7]], axis=0) == reduce_block_23.values[0])
    assert np.allclose(
        np.sum(block_1.values[[3, 5, 6]], axis=0),
        reduce_block_23.values[4],
        rtol=1e-13,
    )

    assert np.all(block_1.values[1] == reduce_block_23.values[1])
    assert np.all(block_1.values[2] == reduce_block_23.values[2])
    assert np.all(block_1.values[4] == reduce_block_23.values[3])

    assert np.all(
        np.sum(block_1.values[[0, 1, 2, 7]], axis=0) == reduce_block_2.values[0]
    )
    assert np.all(np.sum(block_1.values[3:7], axis=0) == reduce_block_2.values[1])

    # check metadata
    assert reduce_block_12.properties == block_1.properties
    assert reduce_block_23.properties == block_1.properties
    assert reduce_block_2.properties == block_1.properties

    samples_12 = Labels(
        names=["s1", "s2"],
        values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]),
    )
    samples_23 = Labels(
        names=["s2", "s3"],
        values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
    )
    samples_2 = Labels(
        names=["s2"],
        values=np.array([[0], [1]]),
    )
    assert reduce_block_12.samples == samples_12
    assert reduce_block_23.samples == samples_23
    assert reduce_block_2.samples == samples_2


def test_reduction_block_multi_properties():
    """Test sum over properties for multiple dimensions simultaneously"""
    block_1 = TensorBlock(
        values=np.array(
            [
                [1, 2, 4],
                [3, 5, 6],
                [-1.3, 26.7, 4.54],
                [3.5, 5.3, 6.87],
                [6.1, 35.2, 44.5],
                [7.3, -7.65, 6.45],
                [11, 276.0, 4.09],
                [33, 55.5, -5.6],
            ]
        ).T,
        samples=Labels(["s"], np.array([[0], [1], [5]])),
        components=[],
        properties=Labels(
            ["p1", "p2", "p3"],
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 1, 1],
                    [0, 1, 0],
                    [2, 1, 1],
                    [1, 1, 1],
                    [1, 0, 0],
                ],
            ),
        ),
    )

    reduce_block_12 = mts.sum_over_properties_block(block_1, property_names="p3")
    reduce_block_23 = mts.sum_over_properties_block(block_1, property_names=["p1"])
    reduce_block_2 = mts.sum_over_properties_block(block_1, property_names=["p1", "p3"])

    assert np.all(
        np.sum(block_1.values[..., :3], axis=-1) == reduce_block_12.values[..., 0]
    )
    assert np.all(
        np.sum(block_1.values[..., 3:5], axis=-1) == reduce_block_12.values[..., 1]
    )
    assert np.all(block_1.values[..., 5] == reduce_block_12.values[..., 4])
    assert np.all(block_1.values[..., 6] == reduce_block_12.values[..., 3])
    assert np.all(block_1.values[..., 7] == reduce_block_12.values[..., 2])

    assert np.all(
        np.sum(block_1.values[..., [0, 7]], axis=-1) == reduce_block_23.values[..., 0]
    )
    assert np.all(
        np.sum(block_1.values[..., [3, 5, 6]], axis=-1)
        == reduce_block_23.values[..., 4]
    )

    assert np.all(block_1.values[..., 1] == reduce_block_23.values[..., 1])
    assert np.all(block_1.values[..., 2] == reduce_block_23.values[..., 2])
    assert np.all(block_1.values[..., 4] == reduce_block_23.values[..., 3])

    assert np.all(
        np.sum(block_1.values[..., [0, 1, 2, 7]], axis=-1)
        == reduce_block_2.values[..., 0]
    )
    assert np.all(
        np.sum(block_1.values[..., 3:7], axis=-1) == reduce_block_2.values[..., 1]
    )
    # check metadata
    assert reduce_block_12.samples == block_1.samples
    assert reduce_block_23.samples == block_1.samples
    assert reduce_block_2.samples == block_1.samples

    properties_12 = Labels(
        names=["p1", "p2"],
        values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]),
    )
    properties_23 = Labels(
        names=["p2", "p3"],
        values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
    )
    properties_2 = Labels(
        names=["p2"],
        values=np.array([[0], [1]]),
    )
    assert reduce_block_12.properties == properties_12
    assert reduce_block_23.properties == properties_23
    assert reduce_block_2.properties == properties_2
