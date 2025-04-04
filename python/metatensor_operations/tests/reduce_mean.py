import os

import numpy as np

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_mean_samples_block():
    tensor_se = metatensor.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))
    tensor_ps = metatensor.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))
    bl1 = tensor_ps[0]

    # check both passing a list and a single string for sample_names
    reduce_tensor_se = metatensor.mean_over_samples(tensor_se, sample_names="atom")
    reduce_tensor_ps = metatensor.mean_over_samples(tensor_ps, sample_names=["atom"])

    assert np.all(
        np.mean(bl1.values[:4], axis=0) == reduce_tensor_ps.block(0).values[0]
    )

    assert np.allclose(
        np.mean(bl1.values[4:10], axis=0),
        reduce_tensor_ps.block(0).values[1],
        rtol=1e-13,
    )

    assert np.all(
        np.mean(bl1.values[22:26], axis=0) == reduce_tensor_ps.block(0).values[5]
    )

    assert np.all(
        np.mean(bl1.values[38:46], axis=0) == reduce_tensor_ps.block(0).values[8]
    )

    assert np.allclose(
        np.mean(bl1.values[46:], axis=0),
        reduce_tensor_ps.block(0).values[9],
        rtol=1e-13,
    )

    # Test the gradients
    gradient = tensor_ps[0].gradient("positions").values

    assert np.allclose(
        np.mean(gradient[[0, 4, 8, 12]], axis=0),
        reduce_tensor_ps.block(0).gradient("positions").values[0],
        rtol=1e-13,
    )
    assert np.allclose(
        np.mean(gradient[[2, 6, 10, 14]], axis=0),
        reduce_tensor_ps.block(0).gradient("positions").values[2],
    )

    assert np.all(
        np.mean(gradient[[3, 7, 11, 15]], axis=0)
        == reduce_tensor_ps.block(0).gradient("positions").values[3]
    )

    assert np.allclose(
        np.mean(gradient[[96, 99, 102]], axis=0),
        reduce_tensor_ps.block(0).gradient("positions").values[40],
        rtol=1e-13,
    )

    assert np.all(
        np.mean(gradient[-1], axis=0)
        == reduce_tensor_ps.block(0).gradient("positions").values[-1]
    )

    # The TensorBlock with key=(8,8,8) has nothing to be averaged over
    selection = {
        "center_type": 8,
        "neighbor_1_type": 8,
        "neighbor_2_type": 8,
    }
    print(tensor_ps.keys.names)
    assert np.allclose(
        tensor_ps.block(selection).values,
        reduce_tensor_ps.block(selection).values,
    )

    for ii, bl2 in enumerate([tensor_se[0], tensor_se[1], tensor_se[2], tensor_se[3]]):
        assert np.all(
            np.mean(bl2.values[:4], axis=0) == reduce_tensor_se.block(ii).values[0]
        )
        assert np.allclose(
            np.mean(bl2.values[26:32], axis=0),
            reduce_tensor_se.block(ii).values[6],
            rtol=1e-13,
        )

        assert np.allclose(
            np.mean(bl2.values[32:38], axis=0),
            reduce_tensor_se.block(ii).values[7],
            rtol=1e-13,
        )
        assert np.allclose(
            np.mean(bl2.values[46:], axis=0),
            reduce_tensor_se.block(ii).values[9],
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

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
    X = TensorMap(keys, [block_1])

    reduce_X_12 = metatensor.mean_over_samples(X, sample_names=["samples3"])
    reduce_X_23 = metatensor.mean_over_samples(X, sample_names="samples1")
    reduce_X_2 = metatensor.mean_over_samples(X, sample_names=["samples1", "samples3"])

    assert np.allclose(
        np.mean(X.block(0).values[:3], axis=0),
        reduce_X_12.block(0).values[0],
        rtol=1e-13,
    )

    assert np.all(
        np.mean(X.block(0).values[3:5], axis=0) == reduce_X_12.block(0).values[1]
    )
    assert np.all(X.block(0).values[5] == reduce_X_12.block(0).values[4])
    assert np.all(X.block(0).values[6] == reduce_X_12.block(0).values[3])
    assert np.all(X.block(0).values[7] == reduce_X_12.block(0).values[2])

    assert np.all(
        np.mean(X.block(0).values[[0, 7]], axis=0) == reduce_X_23.block(0).values[0]
    )
    assert np.allclose(
        np.mean(X.block(0).values[[3, 5, 6]], axis=0),
        reduce_X_23.block(0).values[4],
        rtol=1e-13,
    )

    assert np.all(X.block(0).values[1] == reduce_X_23.block(0).values[1])
    assert np.all(X.block(0).values[2] == reduce_X_23.block(0).values[2])
    assert np.all(X.block(0).values[4] == reduce_X_23.block(0).values[3])

    assert np.all(
        np.mean(X.block(0).values[[0, 1, 2, 7]], axis=0)
        == reduce_X_2.block(0).values[0]
    )
    assert np.all(
        np.mean(X.block(0).values[3:7], axis=0) == reduce_X_2.block(0).values[1]
    )

    # check metadata
    assert reduce_X_12.block(0).properties == X.block(0).properties
    assert reduce_X_23.block(0).properties == X.block(0).properties
    assert reduce_X_2.block(0).properties == X.block(0).properties

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
    assert reduce_X_12.block(0).samples == samples_12
    assert reduce_X_23.block(0).samples == samples_23
    assert reduce_X_2.block(0).samples == samples_2
