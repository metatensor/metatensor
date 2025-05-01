import os

import numpy as np

import metatensor as mts
from metatensor import Labels, TensorBlock


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_mean_samples_block():
    tensor_se = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))
    tensor_ps = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))

    block_ps = tensor_ps[0]
    # check both passing a list and a single string for sample_names
    reduce_block_ps = mts.mean_over_samples_block(block_ps, sample_names=["atom"])

    assert np.all(np.mean(block_ps.values[:4], axis=0) == reduce_block_ps.values[0])

    assert np.allclose(
        np.mean(block_ps.values[4:10], axis=0),
        reduce_block_ps.values[1],
        rtol=1e-13,
    )

    assert np.all(np.mean(block_ps.values[22:26], axis=0) == reduce_block_ps.values[5])

    assert np.all(np.mean(block_ps.values[38:46], axis=0) == reduce_block_ps.values[8])

    assert np.allclose(
        np.mean(block_ps.values[46:], axis=0),
        reduce_block_ps.values[9],
        rtol=1e-13,
    )

    # Test the gradients
    gradient = block_ps.gradient("positions").values

    assert np.allclose(
        np.mean(gradient[[0, 4, 8, 12]], axis=0),
        reduce_block_ps.gradient("positions").values[0],
        rtol=1e-13,
    )
    assert np.allclose(
        np.mean(gradient[[2, 6, 10, 14]], axis=0),
        reduce_block_ps.gradient("positions").values[2],
    )

    assert np.all(
        np.mean(gradient[[3, 7, 11, 15]], axis=0)
        == reduce_block_ps.gradient("positions").values[3]
    )

    assert np.allclose(
        np.mean(gradient[[96, 99, 102]], axis=0),
        reduce_block_ps.gradient("positions").values[40],
        rtol=1e-13,
    )

    assert np.all(
        np.mean(gradient[-1], axis=0)
        == reduce_block_ps.gradient("positions").values[-1]
    )

    # The TensorBlock with key=(8,8,8) has nothing to be averaged over

    for _ii, bl2 in enumerate([tensor_se[0], tensor_se[1], tensor_se[2], tensor_se[3]]):
        reduced_block = mts.mean_over_samples_block(bl2, sample_names="atom")
        assert np.all(np.mean(bl2.values[:4], axis=0) == reduced_block.values[0])
        assert np.allclose(
            np.mean(bl2.values[26:32], axis=0),
            reduced_block.values[6],
            rtol=1e-13,
        )

        assert np.allclose(
            np.mean(bl2.values[32:38], axis=0),
            reduced_block.values[7],
            rtol=1e-13,
        )
        assert np.allclose(
            np.mean(bl2.values[46:], axis=0),
            reduced_block.values[9],
            rtol=1e-13,
        )


def test_reduction_block_two_samples():
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
            ["samples1", "samples2", "samples3"],
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

    reduce_block_12 = mts.mean_over_samples_block(block_1, sample_names=["samples3"])
    reduce_block_23 = mts.mean_over_samples_block(block_1, sample_names="samples1")
    reduce_block_2 = mts.mean_over_samples_block(
        block_1, sample_names=["samples1", "samples3"]
    )

    assert np.allclose(
        np.mean(block_1.values[:3], axis=0),
        reduce_block_12.values[0],
        rtol=1e-13,
    )

    assert np.all(np.mean(block_1.values[3:5], axis=0) == reduce_block_12.values[1])
    assert np.all(block_1.values[5] == reduce_block_12.values[4])
    assert np.all(block_1.values[6] == reduce_block_12.values[3])
    assert np.all(block_1.values[7] == reduce_block_12.values[2])

    assert np.all(np.mean(block_1.values[[0, 7]], axis=0) == reduce_block_23.values[0])
    assert np.allclose(
        np.mean(block_1.values[[3, 5, 6]], axis=0),
        reduce_block_23.values[4],
        rtol=1e-13,
    )

    assert np.all(block_1.values[1] == reduce_block_23.values[1])
    assert np.all(block_1.values[2] == reduce_block_23.values[2])
    assert np.all(block_1.values[4] == reduce_block_23.values[3])

    assert np.all(
        np.mean(block_1.values[[0, 1, 2, 7]], axis=0) == reduce_block_2.values[0]
    )
    assert np.all(np.mean(block_1.values[3:7], axis=0) == reduce_block_2.values[1])

    # check metadata
    assert reduce_block_12.properties == block_1.properties
    assert reduce_block_23.properties == block_1.properties
    assert reduce_block_2.properties == block_1.properties

    samples_12 = Labels(
        names=["samples1", "samples2"],
        values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]),
    )
    samples_23 = Labels(
        names=["samples2", "samples3"],
        values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
    )
    samples_2 = Labels(
        names=["samples2"],
        values=np.array([[0], [1]]),
    )
    assert reduce_block_12.samples == samples_12
    assert reduce_block_23.samples == samples_23
    assert reduce_block_2.samples == samples_2
