import os

import numpy as np


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import metatensor as mts
from metatensor import Labels, TensorBlock, TensorMap
from metatensor.operations import _dispatch

from . import _gradcheck


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


def test_reduction_all_properties():
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

    sum_X = mts.sum_over_properties(X, property_names=["p"])
    mean_X = mts.mean_over_properties(X, property_names=["p"])
    var_X = mts.var_over_properties(X, property_names=["p"])
    std_X = mts.std_over_properties(X, property_names=["p"])

    assert sum_X[0].properties == Labels.single()
    assert mts.equal_metadata(sum_X, mean_X)
    assert mts.equal_metadata(sum_X, std_X)
    assert mts.equal_metadata(mean_X, var_X)

    assert np.all(sum_X[0].values == np.sum(X[0].values, axis=1, keepdims=True))
    assert np.all(mean_X[0].values == np.mean(X[0].values, axis=1, keepdims=True))
    assert np.allclose(std_X[0].values, np.std(X[0].values, axis=1, keepdims=True))
    assert np.allclose(var_X[0].values, np.var(X[0].values, axis=1, keepdims=True))

    if HAS_TORCH:
        X = X.to(arrays="torch")

        sum_X = mts.sum_over_properties(X, property_names=["p"])
        mean_X = mts.mean_over_properties(X, property_names=["p"])
        var_X = mts.var_over_properties(X, property_names=["p"])
        std_X = mts.std_over_properties(X, property_names=["p"])

        assert sum_X[0].properties == Labels.single()
        assert mts.equal_metadata(sum_X, mean_X)
        assert mts.equal_metadata(sum_X, std_X)
        assert mts.equal_metadata(mean_X, var_X)

        assert torch.all(
            sum_X[0].values == torch.sum(X[0].values, axis=1, keepdims=True)
        )
        assert torch.all(
            mean_X[0].values == torch.mean(X[0].values, axis=1, keepdims=True)
        )
        assert torch.allclose(
            std_X[0].values, torch.std(X[0].values, correction=0, axis=1, keepdims=True)
        )
        assert torch.allclose(
            var_X[0].values, torch.var(X[0].values, correction=0, axis=1, keepdims=True)
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


def test_zeros_property_block():
    block = TensorBlock(
        values=np.zeros([1, 0]),
        properties=Labels(["p"], np.empty((0, 1))),
        samples=Labels(["s"], np.zeros([1, 1], dtype=int)),
        components=[],
    )

    result_block = TensorBlock(
        values=np.zeros([1, 0]),
        properties=Labels([], np.empty((0, 0))),
        samples=Labels(["s"], np.zeros([1, 1], dtype=int)),
        components=[],
    )

    tensor = TensorMap(Labels.single(), [block])
    result_tensor = TensorMap(Labels.single(), [result_block])

    tensor_sum = mts.sum_over_properties(tensor, "p")
    tensor_mean = mts.mean_over_properties(tensor, "p")
    tensor_std = mts.std_over_properties(tensor, "p")
    tensor_var = mts.var_over_properties(tensor, "p")

    assert mts.equal(result_tensor, tensor_sum)
    assert mts.equal(result_tensor, tensor_mean)
    assert mts.equal(result_tensor, tensor_var)
    assert mts.equal(result_tensor, tensor_std)

    block = TensorBlock(
        values=np.zeros([1, 0]),
        properties=Labels(["p_1", "p_2"], np.empty((0, 1))),
        samples=Labels(["s"], np.zeros([1, 1], dtype=int)),
        components=[],
    )

    result_block = TensorBlock(
        values=np.zeros([1, 0]),
        properties=Labels(["p_2"], np.empty((0, 1))),
        samples=Labels(["s"], np.zeros([1, 1], dtype=int)),
        components=[],
    )

    tensor = TensorMap(Labels.single(), [block])
    result_tensor = TensorMap(Labels.single(), [result_block])

    tensor_sum = mts.sum_over_properties(tensor, "p_1")
    tensor_mean = mts.mean_over_properties(tensor, "p_1")
    tensor_std = mts.std_over_properties(tensor, "p_1")
    tensor_var = mts.var_over_properties(tensor, "p_1")

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


def test_zeros_property_block_gradient():
    block = TensorBlock(
        values=np.array([[1, 2, 4], [3, 5, 6], [-1.3, 26.7, 4.54], [3.5, 5.3, 6.87]]).T,
        samples=Labels(["s"], np.array([[0], [1], [5]])),
        components=[],
        properties=Labels(
            ["p_1", "p_2"],
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        ),
    )

    block.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.zeros((0, 4)),
            samples=Labels(["sample", "g"], np.empty((0, 2))),
            components=[],
            properties=block.properties,
        ),
    )

    sum_block = TensorBlock(
        values=np.array([[-0.3, 28.7, 8.54], [6.5, 10.3, 12.87]]).T,
        samples=Labels(["s"], np.array([[0], [1], [5]])),
        components=[],
        properties=Labels.range("p_2", 2),
    )

    sum_block.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.zeros((0, 2)),
            samples=Labels(["sample", "g"], np.empty((0, 2))),
            components=[],
            properties=sum_block.properties,
        ),
    )

    tensor = TensorMap(Labels.single(), [block])
    tensor_sum_result = TensorMap(Labels.single(), [sum_block])

    tensor_sum = mts.sum_over_properties(tensor, "p_1")
    tensor_mean = mts.mean_over_properties(tensor, "p_1")
    tensor_std = mts.std_over_properties(tensor, "p_1")
    tensor_var = mts.var_over_properties(tensor, "p_1")

    assert mts.allclose(tensor_sum_result, tensor_sum, atol=1e-14)
    assert mts.equal_metadata(tensor_sum, tensor_mean)
    assert mts.equal_metadata(tensor_sum, tensor_var)
    assert mts.equal_metadata(tensor_sum, tensor_std)


def tensor_for_finite_differences(array, parameter) -> TensorMap:
    """
    Creates a TensorMap from a set of cartesian vectors according to the function:

    .. math::

        f(x, y, z) = x^3 + y^3 + z^3

        \\nabla f = (3x^2, 3y^2, 3z^2)

    The gradients are stored with the given ``parameter`` name.
    """
    n_samples = array.shape[0]
    n_properties = array.shape[-1]
    assert array.shape == (n_samples, 3, n_properties)

    values = _dispatch.sum(array**3, axis=1)
    values_grad = 3 * array**2

    block = mts.TensorBlock(
        values,
        Labels.range("s", 10),
        [],
        Labels(
            ["p_1", "p_2"], np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
        ),
    )
    block.add_gradient(
        parameter=parameter,
        gradient=TensorBlock(
            values=values_grad.reshape(n_samples, 3, n_properties),
            samples=Labels.range("sample", len(values)),
            components=[Labels.range("xyz", 3)],
            properties=block.properties,
        ),
    )

    return TensorMap(Labels.range("_", 1), [block])


def test_finite_difference():
    def sum(array):
        tensor = tensor_for_finite_differences(array, parameter="g")
        return mts.sum_over_properties(tensor, "p_1")

    def mean(array):
        tensor = tensor_for_finite_differences(array, parameter="g")
        return mts.mean_over_properties(tensor, "p_1")

    def var(array):
        tensor = tensor_for_finite_differences(array, parameter="g")
        return mts.var_over_properties(tensor, "p_1")

    def std(array):
        tensor = tensor_for_finite_differences(array, parameter="g")
        return mts.std_over_properties(tensor, "p_1")

    rng = np.random.default_rng(seed=123456)
    array = rng.random((10, 3, 6))
    _gradcheck.check_finite_differences(sum, array, parameter="g")
    _gradcheck.check_finite_differences(mean, array, parameter="g")
    _gradcheck.check_finite_differences(var, array, parameter="g")
    _gradcheck.check_finite_differences(std, array, parameter="g")
