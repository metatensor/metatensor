import os

import numpy as np

import metatensor as mts
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_std_samples_block():
    tensor_se = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))
    tensor_ps = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))
    tensor_se = mts.remove_gradients(tensor_se)

    bl1 = tensor_ps[0]

    # check both passing a list and a single string for sample_names
    reduce_tensor_se = mts.std_over_samples(tensor_se, sample_names="atom")
    reduce_tensor_ps = mts.std_over_samples(tensor_ps, sample_names=["atom"])

    assert np.allclose(
        np.std(bl1.values[:4], axis=0),
        reduce_tensor_ps.block(0).values[0],
        rtol=1e-13,
    )

    assert np.allclose(
        np.std(bl1.values[4:10], axis=0),
        reduce_tensor_ps.block(0).values[1],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[22:26], axis=0),
        reduce_tensor_ps.block(0).values[5],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[38:46], axis=0),
        reduce_tensor_ps.block(0).values[8],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[46:], axis=0),
        reduce_tensor_ps.block(0).values[9],
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
        reduce_tensor_ps.block(0).gradient("positions").values[0],
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
        reduce_tensor_ps.block(0).gradient("positions").values[2],
    )

    XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[3, 7, 11, 15])
    assert np.allclose(
        (
            np.mean(XdX, axis=0)
            - np.mean(bl1.values[:4], axis=0)
            * np.mean(gr1.values[[3, 7, 11, 15]], axis=0)
        )
        / np.std(bl1.values[:4], axis=0),
        reduce_tensor_ps.block(0).gradient("positions").values[3],
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
        reduce_tensor_ps.block(0).gradient("positions").values[40],
        rtol=1e-13,
    )

    # The TensorBlock with key=(8,8,8) has nothing to be averaged over
    values = reduce_tensor_ps.block(
        center_type=8, neighbor_1_type=8, neighbor_2_type=8
    ).values
    assert np.allclose(
        np.zeros(values.shape),
        values,
    )

    for ii, bl2 in enumerate([tensor_se[0], tensor_se[1], tensor_se[2], tensor_se[3]]):
        assert np.allclose(
            np.std(bl2.values[:4], axis=0),
            reduce_tensor_se.block(ii).values[0],
            rtol=1e-13,
        )
        assert np.allclose(
            np.std(bl2.values[26:32], axis=0),
            reduce_tensor_se.block(ii).values[6],
            rtol=1e-13,
        )
        assert np.allclose(
            np.std(bl2.values[32:38], axis=0),
            reduce_tensor_se.block(ii).values[7],
            rtol=1e-13,
        )
        assert np.allclose(
            np.std(bl2.values[46:], axis=0),
            reduce_tensor_se.block(ii).values[9],
            rtol=1e-13,
        )


def test_std_properties_block():
    tensor_se = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))
    tensor_ps = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))
    tensor_se = mts.remove_gradients(tensor_se)

    bl1 = tensor_ps[0]

    # check both passing a list and a single string for property_names
    reduce_tensor_se = mts.std_over_properties(tensor_se, property_names="n")
    reduce_tensor_ps = mts.std_over_properties(tensor_ps, property_names=["l"])

    assert np.allclose(
        np.std(bl1.values[..., ::16], axis=-1),
        reduce_tensor_ps.block(0).values[..., 0],
        rtol=1e-13,
    )

    assert np.allclose(
        np.std(bl1.values[..., 1::16], axis=-1),
        reduce_tensor_ps.block(0).values[..., 1],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[..., 5::16], axis=-1),
        reduce_tensor_ps.block(0).values[..., 5],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[..., 8::16], axis=-1),
        reduce_tensor_ps.block(0).values[..., 8],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[..., 15::16], axis=-1),
        reduce_tensor_ps.block(0).values[..., 15],
        rtol=1e-13,
    )

    # Test the gradients
    gr1 = tensor_ps[0].gradient("positions")
    sample_idx = gr1.samples["sample"]
    other_dims = len(gr1.values.shape) - 2
    assert np.allclose(
        (
            (
                gr1.values[..., ::16]
                * bl1.values[:, ::16][sample_idx].reshape(
                    (len(sample_idx),) + (1,) * other_dims + (-1,)
                )
            ).mean(axis=-1)
            - gr1.values[..., ::16].mean(axis=-1)
            * bl1.values[:, ::16].mean(axis=-1)[sample_idx][:, None]
        )
        / bl1.values[:, ::16].std(axis=-1)[sample_idx][:, None],
        reduce_tensor_ps[0].gradient("positions").values[..., 0],
    )

    assert np.allclose(
        (
            (
                gr1.values[..., 1::16]
                * bl1.values[:, 1::16][sample_idx].reshape(
                    (len(sample_idx),) + (1,) * other_dims + (-1,)
                )
            ).mean(axis=-1)
            - gr1.values[..., 1::16].mean(axis=-1)
            * bl1.values[:, 1::16].mean(axis=-1)[sample_idx][:, None]
        )
        / bl1.values[:, 1::16].std(axis=-1)[sample_idx][:, None],
        reduce_tensor_ps[0].gradient("positions").values[..., 1],
    )
    assert np.allclose(
        (
            (
                gr1.values[..., 5::16]
                * bl1.values[:, 5::16][sample_idx].reshape(
                    (len(sample_idx),) + (1,) * other_dims + (-1,)
                )
            ).mean(axis=-1)
            - gr1.values[..., 5::16].mean(axis=-1)
            * bl1.values[:, 5::16].mean(axis=-1)[sample_idx][:, None]
        )
        / bl1.values[:, 5::16].std(axis=-1)[sample_idx][:, None],
        reduce_tensor_ps[0].gradient("positions").values[..., 5],
    )

    assert np.allclose(
        (
            (
                gr1.values[..., 15::16]
                * bl1.values[:, 15::16][sample_idx].reshape(
                    (len(sample_idx),) + (1,) * other_dims + (-1,)
                )
            ).mean(axis=-1)
            - gr1.values[..., 15::16].mean(axis=-1)
            * bl1.values[:, 15::16].mean(axis=-1)[sample_idx][:, None]
        )
        / bl1.values[:, 15::16].std(axis=-1)[sample_idx][:, None],
        reduce_tensor_ps[0].gradient("positions").values[..., 15],
    )

    for ii, bl2 in enumerate([tensor_se[0], tensor_se[1], tensor_se[2], tensor_se[3]]):
        assert np.allclose(
            np.std(bl2.values, axis=-1, keepdims=True),
            reduce_tensor_se.block(ii).values,
            rtol=1e-13,
        )


def test_reduction_block_multi_samples():
    """Test std over samples for multiple dimensions simultaneously"""
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

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
    X = TensorMap(keys, [block_1])

    reduce_X_12 = mts.std_over_samples(X, sample_names=["s3"])
    reduce_X_23 = mts.std_over_samples(X, sample_names="s1")
    reduce_X_2 = mts.std_over_samples(X, sample_names=["s1", "s3"])

    assert np.allclose(
        np.std(X.block(0).values[:3], axis=0),
        reduce_X_12.block(0).values[0],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(X.block(0).values[3:5], axis=0),
        reduce_X_12.block(0).values[1],
        rtol=1e-13,
    )
    assert np.all(np.array([0.0]) == reduce_X_12.block(0).values[4])
    assert np.all(np.array([0.0]) == reduce_X_12.block(0).values[3])
    assert np.all(np.array([0.0]) == reduce_X_12.block(0).values[2])

    assert np.all(
        np.std(X.block(0).values[[0, 7]], axis=0) == reduce_X_23.block(0).values[0]
    )
    assert np.allclose(
        np.std(X.block(0).values[[3, 5, 6]], axis=0),
        reduce_X_23.block(0).values[4],
        rtol=1e-13,
    )

    assert np.all(np.array([0.0]) == reduce_X_23.block(0).values[1])
    assert np.all(np.array([0.0]) == reduce_X_23.block(0).values[2])
    assert np.all(np.array([0.0]) == reduce_X_23.block(0).values[3])

    assert np.allclose(
        np.std(X.block(0).values[[0, 1, 2, 7]], axis=0),
        reduce_X_2.block(0).values[0],
        rtol=1e-13,
    )
    assert np.all(
        np.std(X.block(0).values[3:7], axis=0) == reduce_X_2.block(0).values[1]
    )

    # check metadata
    assert reduce_X_12.block(0).properties == X.block(0).properties
    assert reduce_X_23.block(0).properties == X.block(0).properties
    assert reduce_X_2.block(0).properties == X.block(0).properties

    samples12 = Labels(
        names=["s1", "s2"],
        values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]),
    )
    samples23 = Labels(
        names=["s2", "s3"],
        values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
    )
    samples2 = Labels(names=["s2"], values=np.array([[0], [1]]))
    assert reduce_X_12.block(0).samples == samples12
    assert reduce_X_23.block(0).samples == samples23
    assert reduce_X_2.block(0).samples == samples2


def test_reduction_block_multi_properties():
    """Test std over properties for multiple dimensions simultaneously"""
    block_1 = TensorBlock(
        values=np.array(
            [
                [1.0, 3.0, -1.3, 3.5, 6.1, 7.3, 11.0, 33.0],
                [2.0, 5.0, 26.7, 5.3, 35.2, -7.65, 276.0, 55.5],
                [4.0, 6.0, 4.54, 6.87, 44.5, 6.45, 4.09, -5.6],
            ]
        ),
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

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
    X = TensorMap(keys, [block_1])

    reduce_X_12 = mts.std_over_properties(X, property_names=["p3"])
    reduce_X_23 = mts.std_over_properties(X, property_names="p1")
    reduce_X_2 = mts.std_over_properties(X, property_names=["p1", "p3"])

    assert np.allclose(
        np.std(X.block(0).values[..., :3], axis=-1),
        reduce_X_12.block(0).values[..., 0],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(X.block(0).values[..., 3:5], axis=-1),
        reduce_X_12.block(0).values[..., 1],
        rtol=1e-13,
    )
    assert np.all(np.array([0.0]) == reduce_X_12.block(0).values[..., 4])
    assert np.all(np.array([0.0]) == reduce_X_12.block(0).values[..., 3])
    assert np.all(np.array([0.0]) == reduce_X_12.block(0).values[..., 2])

    assert np.all(
        np.std(X.block(0).values[..., [0, 7]], axis=-1)
        == reduce_X_23.block(0).values[..., 0]
    )
    assert np.allclose(
        np.std(X.block(0).values[..., [3, 5, 6]], axis=-1),
        reduce_X_23.block(0).values[..., 4],
        rtol=1e-13,
    )

    assert np.all(np.array([0.0]) == reduce_X_23.block(0).values[..., 1])
    assert np.all(np.array([0.0]) == reduce_X_23.block(0).values[..., 2])
    assert np.all(np.array([0.0]) == reduce_X_23.block(0).values[..., 3])

    assert np.allclose(
        np.std(X.block(0).values[..., [0, 1, 2, 7]], axis=-1),
        reduce_X_2.block(0).values[..., 0],
        rtol=1e-13,
    )
    assert np.all(
        np.std(X.block(0).values[..., 3:7], axis=-1)
        == reduce_X_2.block(0).values[..., 1]
    )

    # check metadata
    assert reduce_X_12.block(0).samples == X.block(0).samples
    assert reduce_X_23.block(0).samples == X.block(0).samples
    assert reduce_X_2.block(0).samples == X.block(0).samples

    properties12 = Labels(
        names=["p1", "p2"],
        values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]),
    )
    properties23 = Labels(
        names=["p2", "p3"],
        values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
    )
    properties2 = Labels(names=["p2"], values=np.array([[0], [1]]))
    assert reduce_X_12.block(0).properties == properties12
    assert reduce_X_23.block(0).properties == properties23
    assert reduce_X_2.block(0).properties == properties2


def test_reduction_samples_single():
    """
    Test std_over_samples when there is only one element per sample class being
    reduced (i.e. each sample is treated independently)
    """
    block_1 = TensorBlock(
        values=np.array([[1, 2, 4], [3, 5, 6], [-1.3, 26.7, 4.54]]),
        samples=Labels(["s1", "s2"], np.array([[0, 0], [1, 1], [2, 2]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1], [5]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7.8]]),
            samples=Labels(["sample", "g"], np.array([[0, 0], [1, 1], [2, 2]])),
            components=[],
            properties=block_1.properties,
        ),
    )

    keys = Labels(names=["key_1"], values=np.array([[0]]))
    X = TensorMap(keys, [block_1])

    add_X = mts.sum_over_samples(X, sample_names=["s1"])
    mean_X = mts.mean_over_samples(X, sample_names=["s1"])
    var_X = mts.var_over_samples(X, sample_names=["s1"])
    std_X = mts.std_over_samples(X, sample_names=["s1"])

    assert np.all(X[0].values == add_X[0].values)
    assert np.all(X[0].values == mean_X[0].values)
    assert mts.equal(add_X, mean_X)
    assert mts.equal_metadata(add_X, var_X)
    assert mts.equal_metadata(mean_X, std_X)

    assert np.all(np.zeros((3, 3)) == std_X[0].values)
    assert mts.equal(var_X, std_X)

    # Gradients
    grad_sample_label = Labels(
        names=["sample", "g"],
        values=np.array([[0, 0], [1, 1], [2, 2]]),
    )

    assert std_X[0].gradient("g").samples == grad_sample_label
    assert np.all(X[0].gradient("g").values == add_X[0].gradient("g").values)
    assert np.all(X[0].gradient("g").values == mean_X[0].gradient("g").values)
    assert np.all(np.zeros((3, 3)) == std_X[0].gradient("g").values)
    assert np.all(np.zeros((3, 3)) == var_X[0].gradient("g").values)


def test_reduction_properties_single():
    """
    Test std_over_properties when there is only one element per property class being
    reduced (i.e. each property is treated independently)
    """
    block_1 = TensorBlock(
        values=np.array([[1, 2, 4], [3, 5, 6], [-1.3, 26.7, 4.54]]).T,
        samples=Labels(["p"], np.array([[0], [1], [5]])),
        components=[],
        properties=Labels(["s1", "s2"], np.array([[0, 0], [1, 1], [2, 2]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7.8]]).T,
            samples=Labels(["sample"], np.array([[0], [1], [2]])),
            components=[],
            properties=block_1.properties,
        ),
    )

    keys = Labels(names=["key_1"], values=np.array([[0]]))
    X = TensorMap(keys, [block_1])

    add_X = mts.sum_over_properties(X, property_names=["s1"])
    mean_X = mts.mean_over_properties(X, property_names=["s1"])
    var_X = mts.var_over_properties(X, property_names=["s1"])
    std_X = mts.std_over_properties(X, property_names=["s1"])

    assert np.all(X[0].values == add_X[0].values)
    assert np.all(X[0].values == mean_X[0].values)
    assert mts.equal(add_X, mean_X)
    assert mts.equal_metadata(add_X, var_X)
    assert mts.equal_metadata(mean_X, std_X)

    assert np.all(np.zeros((3, 3)) == std_X[0].values)
    assert mts.equal(var_X, std_X)

    # Gradients
    grad_properties_label = Labels(["s2"], np.array([[0], [1], [2]]))

    assert std_X[0].gradient("g").properties == grad_properties_label
    assert np.all(X[0].gradient("g").values == add_X[0].gradient("g").values)
    assert np.all(X[0].gradient("g").values == mean_X[0].gradient("g").values)
    assert np.all(np.zeros((3, 3)) == std_X[0].gradient("g").values)
    assert np.all(np.zeros((3, 3)) == var_X[0].gradient("g").values)


def get_XdX(block, gradient, der_index):
    XdX = []
    for ig in der_index:
        idx = gradient.samples[ig][0]
        XdX.append(block.values[idx] * gradient.values[ig])
    return np.stack(XdX)
