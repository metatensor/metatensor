import os

import numpy as np

import metatensor as mts
from metatensor import Labels, TensorBlock


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_std_samples_block():
    tensor_se = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))
    tensor_ps = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))
    tensor_se = mts.remove_gradients(tensor_se)

    bl1 = tensor_ps[0]

    # check both passing a list and a single string for sample_names
    reduce_bl_ps = mts.std_over_samples_block(bl1, sample_names=["atom"])

    assert np.allclose(
        np.std(bl1.values[:4], axis=0),
        reduce_bl_ps.values[0],
        rtol=1e-13,
    )

    assert np.allclose(
        np.std(bl1.values[4:10], axis=0),
        reduce_bl_ps.values[1],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[22:26], axis=0),
        reduce_bl_ps.values[5],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[38:46], axis=0),
        reduce_bl_ps.values[8],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[46:], axis=0),
        reduce_bl_ps.values[9],
        rtol=1e-13,
    )

    # Test the gradients
    gr1 = tensor_ps[0].gradient("positions")

    XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[0, 4, 8, 12])
    assert np.allclose(
        (
            np.mean(XdX, axis=0)
            - np.mean(bl1.values[:4], axis=0)
            * np.mean(gr1.values[[0, 4, 8, 12]], axis=0)
        )
        / np.std(bl1.values[:4], axis=0),
        reduce_bl_ps.gradient("positions").values[0],
        rtol=1e-13,
    )

    XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[2, 6, 10, 14])
    assert np.allclose(
        (
            np.mean(XdX, axis=0)
            - np.mean(bl1.values[:4], axis=0)
            * np.mean(gr1.values[[2, 6, 10, 14]], axis=0)
        )
        / np.std(bl1.values[:4], axis=0),
        reduce_bl_ps.gradient("positions").values[2],
    )

    XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[3, 7, 11, 15])
    assert np.allclose(
        (
            np.mean(XdX, axis=0)
            - np.mean(bl1.values[:4], axis=0)
            * np.mean(gr1.values[[3, 7, 11, 15]], axis=0)
        )
        / np.std(bl1.values[:4], axis=0),
        reduce_bl_ps.gradient("positions").values[3],
    )

    XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[96, 99, 102])
    idx = [
        i
        for i in range(len(bl1.samples))
        if bl1.samples[i][0] == bl1.samples[gr1.samples[96][0]][0]
    ]

    assert np.allclose(
        (
            np.mean(XdX, axis=0)
            - np.mean(bl1.values[idx], axis=0)
            * np.mean(gr1.values[[96, 99, 102]], axis=0)
        )
        / np.std(bl1.values[idx], axis=0),
        reduce_bl_ps.gradient("positions").values[40],
        rtol=1e-13,
    )

    # The TensorBlock with key=(8,8,8) has nothing to be averaged over
    values = mts.std_over_samples_block(
        tensor_ps.block(center_type=8, neighbor_1_type=8, neighbor_2_type=8),
        sample_names="atom",
    ).values
    assert np.allclose(
        np.zeros(values.shape),
        values,
    )

    for _ii, bl2 in enumerate([tensor_se[0], tensor_se[1], tensor_se[2], tensor_se[3]]):
        reduced_block = mts.std_over_samples_block(bl2, sample_names="atom")
        assert np.allclose(
            np.std(bl2.values[:4], axis=0),
            reduced_block.values[0],
            rtol=1e-13,
        )
        assert np.allclose(
            np.std(bl2.values[26:32], axis=0),
            reduced_block.values[6],
            rtol=1e-13,
        )
        assert np.allclose(
            np.std(bl2.values[32:38], axis=0),
            reduced_block.values[7],
            rtol=1e-13,
        )
        assert np.allclose(
            np.std(bl2.values[46:], axis=0),
            reduced_block.values[9],
            rtol=1e-13,
        )


def test_reduction_block_two_samples():
    block = TensorBlock(
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
            ["s_1", "s_2", "s_3"],
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

    reduce_block_12 = mts.std_over_samples_block(block, sample_names=["s_3"])
    reduce_block_23 = mts.std_over_samples_block(block, sample_names="s_1")
    reduce_block_2 = mts.std_over_samples_block(block, sample_names=["s_1", "s_3"])

    assert np.allclose(
        np.std(block.values[:3], axis=0),
        reduce_block_12.values[0],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(block.values[3:5], axis=0),
        reduce_block_12.values[1],
        rtol=1e-13,
    )
    assert np.all(np.array([0.0]) == reduce_block_12.values[4])
    assert np.all(np.array([0.0]) == reduce_block_12.values[3])
    assert np.all(np.array([0.0]) == reduce_block_12.values[2])

    assert np.all(np.std(block.values[[0, 7]], axis=0) == reduce_block_23.values[0])
    assert np.allclose(
        np.std(block.values[[3, 5, 6]], axis=0),
        reduce_block_23.values[4],
        rtol=1e-13,
    )

    assert np.all(np.array([0.0]) == reduce_block_23.values[1])
    assert np.all(np.array([0.0]) == reduce_block_23.values[2])
    assert np.all(np.array([0.0]) == reduce_block_23.values[3])

    assert np.allclose(
        np.std(block.values[[0, 1, 2, 7]], axis=0),
        reduce_block_2.values[0],
        rtol=1e-13,
    )
    assert np.all(np.std(block.values[3:7], axis=0) == reduce_block_2.values[1])

    # check metadata
    assert reduce_block_12.properties == block.properties
    assert reduce_block_23.properties == block.properties
    assert reduce_block_2.properties == block.properties

    samples_12 = Labels(
        names=["s_1", "s_2"],
        values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]),
    )
    samples_23 = Labels(
        names=["s_2", "s_3"],
        values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
    )
    samples_2 = Labels(
        names=["s_2"],
        values=np.array([[0], [1]]),
    )
    assert reduce_block_12.samples == samples_12
    assert reduce_block_23.samples == samples_23
    assert reduce_block_2.samples == samples_2


def get_XdX(block, gradient, der_index):
    XdX = []
    for ig in der_index:
        idx = gradient.samples[ig][0]
        XdX.append(block.values[idx] * gradient.values[ig])
    return np.stack(XdX)


def test_issue_902():
    block_f64 = mts.block_from_array(
        np.array(
            [[-0.27714047], [-0.27715549], [-0.27721998], [-0.27707845]],
            dtype=np.float64,
        )
    )
    reduced = mts.std_over_samples_block(
        block_f64, sample_names=block_f64.samples.names
    )
    assert not np.isnan(reduced.values)

    block_f32 = block_f64.to(dtype=np.float32)
    reduced = mts.std_over_samples_block(
        block_f32, sample_names=block_f32.samples.names
    )
    assert not np.isnan(reduced.values)
