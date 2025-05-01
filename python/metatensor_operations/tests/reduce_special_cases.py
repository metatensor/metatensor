import os

import numpy as np


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import metatensor as mts
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_reduction_all_samples():
    block_1 = TensorBlock(
        values=np.array(
            [
                [1, 2, 4],
                [3, 5, 6],
                [-1.3, 26.7, 4.54],
            ]
        ),
        samples=Labels.range("s", 3),
        components=[],
        properties=Labels(["p"], np.array([[0], [1], [5]])),
    )
    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
    X = TensorMap(keys, [block_1])

    sum_X = mts.sum_over_samples(X, sample_names=["s"])
    mean_X = mts.mean_over_samples(X, sample_names=["s"])
    var_X = mts.var_over_samples(X, sample_names=["s"])
    std_X = mts.std_over_samples(X, sample_names=["s"])

    assert sum_X[0].samples == Labels.single()
    assert mts.equal_metadata(sum_X, mean_X)
    assert mts.equal_metadata(sum_X, std_X)
    assert mts.equal_metadata(mean_X, var_X)

    assert np.all(sum_X[0].values == np.sum(X[0].values, axis=0))
    assert np.all(mean_X[0].values == np.mean(X[0].values, axis=0))
    assert np.allclose(std_X[0].values, np.std(X[0].values, axis=0))
    assert np.allclose(var_X[0].values, np.var(X[0].values, axis=0))

    if HAS_TORCH:
        X = X.to(arrays="torch")

        sum_X = mts.sum_over_samples(X, sample_names=["s"])
        mean_X = mts.mean_over_samples(X, sample_names=["s"])
        var_X = mts.var_over_samples(X, sample_names=["s"])
        std_X = mts.std_over_samples(X, sample_names=["s"])

        assert sum_X[0].samples == Labels.single()
        assert mts.equal_metadata(sum_X, mean_X)
        assert mts.equal_metadata(sum_X, std_X)
        assert mts.equal_metadata(mean_X, var_X)

        assert torch.all(sum_X[0].values == torch.sum(X[0].values, axis=0))
        assert torch.all(mean_X[0].values == torch.mean(X[0].values, axis=0))
        assert torch.allclose(
            std_X[0].values, torch.std(X[0].values, correction=0, axis=0)
        )
        assert torch.allclose(
            var_X[0].values, torch.var(X[0].values, correction=0, axis=0)
        )


def test_zeros_sample_block():
    block = TensorBlock(
        values=np.zeros([0, 1]),
        properties=Labels(["p"], np.zeros([1, 1], dtype=int)),
        samples=Labels(["s"], np.empty((0, 1))),
        components=[],
    )

    result_block = TensorBlock(
        values=np.zeros([0, 1]),
        properties=Labels(["p"], np.zeros([1, 1], dtype=int)),
        samples=Labels([], np.empty((0, 0))),
        components=[],
    )

    tensor = TensorMap(Labels.single(), [block])
    result_tensor = TensorMap(Labels.single(), [result_block])

    tensor_sum = mts.sum_over_samples(tensor, "s")
    tensor_mean = mts.mean_over_samples(tensor, "s")
    tensor_std = mts.std_over_samples(tensor, "s")
    tensor_var = mts.var_over_samples(tensor, "s")

    assert mts.equal(result_tensor, tensor_sum)
    assert mts.equal(result_tensor, tensor_mean)
    assert mts.equal(result_tensor, tensor_var)
    assert mts.equal(result_tensor, tensor_std)

    block = TensorBlock(
        values=np.zeros([0, 1]),
        properties=Labels(["p"], np.zeros([1, 1], dtype=int)),
        samples=Labels(["s_1", "s_2"], np.empty((0, 1))),
        components=[],
    )

    result_block = TensorBlock(
        values=np.zeros([0, 1]),
        properties=Labels(["p"], np.zeros([1, 1], dtype=int)),
        samples=Labels(["s_2"], np.empty((0, 1))),
        components=[],
    )

    tensor = TensorMap(Labels.single(), [block])
    result_tensor = TensorMap(Labels.single(), [result_block])

    tensor_sum = mts.sum_over_samples(tensor, "s_1")
    tensor_mean = mts.mean_over_samples(tensor, "s_1")
    tensor_std = mts.std_over_samples(tensor, "s_1")
    tensor_var = mts.var_over_samples(tensor, "s_1")

    assert mts.equal(result_tensor, tensor_sum)
    assert mts.equal(result_tensor, tensor_mean)
    assert mts.equal(result_tensor, tensor_var)
    assert mts.equal(result_tensor, tensor_std)


def test_zeros_sample_block_gradient():
    block = TensorBlock(
        values=np.array([[1, 2, 4], [3, 5, 6], [-1.3, 26.7, 4.54], [3.5, 5.3, 6.87]]),
        samples=Labels(
            ["s_1", "s_2"],
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        ),
        components=[],
        properties=Labels(["p"], np.array([[0], [1], [5]])),
    )

    block.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.zeros((0, 3)),
            samples=Labels(["sample", "g"], np.empty((0, 2))),
            components=[],
            properties=block.properties,
        ),
    )

    sum_block = TensorBlock(
        values=np.array([[-0.3, 28.7, 8.54], [6.5, 10.3, 12.87]]),
        samples=Labels.range("s_2", 2),
        components=[],
        properties=Labels(["p"], np.array([[0], [1], [5]])),
    )

    sum_block.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.zeros((0, 3)),
            samples=Labels(["sample", "g"], np.empty((0, 2))),
            components=[],
            properties=sum_block.properties,
        ),
    )

    tensor = TensorMap(Labels.single(), [block])
    tensor_sum_result = TensorMap(Labels.single(), [sum_block])

    tensor_sum = mts.sum_over_samples(tensor, "s_1")
    tensor_mean = mts.mean_over_samples(tensor, "s_1")
    tensor_std = mts.std_over_samples(tensor, "s_1")
    tensor_var = mts.var_over_samples(tensor, "s_1")

    assert mts.allclose(tensor_sum_result, tensor_sum, atol=1e-14)
    assert mts.equal_metadata(tensor_sum, tensor_mean)
    assert mts.equal_metadata(tensor_sum, tensor_var)
    assert mts.equal_metadata(tensor_sum, tensor_std)
