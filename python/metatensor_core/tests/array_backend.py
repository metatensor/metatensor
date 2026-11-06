import os

import numpy as np
import pytest
from numpy.testing import assert_equal

import metatensor
from metatensor._status import catch_exceptions


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "metatensor-core",
    "tests",
    "data.mts",
)

# Arrays with a custom (non-Python) origin. The data is still allocated by numpy, but
# metatensor has to go through DLPack to access it.
TEST_ORIGIN = metatensor._data._array._register_origin("python.test-origin")


@catch_exceptions
def get_test_origin(this, origin):
    origin[0] = TEST_ORIGIN


GET_TEST_ORIGIN = metatensor._data._array._cast_to_ctype_functype(
    get_test_origin, "origin"
)


def _keep_test_origin(name, function):
    # arrays created by metatensor-core from an existing array (e.g. when computing the
    # union of two labels) should use the same origin
    @catch_exceptions
    def wrapper(*args):
        status = function(*args)
        assert status == 0
        set_test_origin(args[-1][0])

    return metatensor._data._array._cast_to_ctype_functype(wrapper, name)


TEST_ORIGIN_CALLBACKS = {
    "create": _keep_test_origin("create", metatensor._data._array._mts_array_create),
    "copy": _keep_test_origin("copy", metatensor._data._array._mts_array_copy),
    "from_dlpack": _keep_test_origin(
        "from_dlpack", metatensor._data._array._mts_array_from_dlpack
    ),
}


def set_test_origin(mts_array):
    mts_array.origin = GET_TEST_ORIGIN
    for name, callback in TEST_ORIGIN_CALLBACKS.items():
        setattr(mts_array, name, callback)


@catch_exceptions
def create_test_array(shape_ptr, shape_count, dtype, array):
    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    data = np.zeros(shape, dtype=metatensor._data.dlpack_dtype_to_numpy(dtype))
    mts_array = metatensor._data.create_mts_array(data)
    set_test_origin(mts_array)

    array[0] = mts_array


@pytest.fixture
def reference():
    return metatensor.load(DATA_PATH)


def check_array_origin(data, expected_origin):
    mts_array = data._raw_values
    origin = metatensor._data.data_origin(mts_array)
    assert metatensor._data.data_origin_name(origin) == expected_origin

    if mts_array.destroy:
        mts_array.destroy(mts_array.ptr)


def check_array_type(array, array_backend):
    if array_backend == "numpy":
        assert isinstance(array, np.ndarray)
    elif array_backend == "torch":
        assert isinstance(array, torch.Tensor)
    else:
        raise ValueError(f"unknown array backend: {array_backend}")


def array_to_backend(array, array_backend):
    if array_backend == "numpy":
        return np.array(array)
    elif array_backend == "torch":
        return torch.tensor(array)
    else:
        raise ValueError(f"unknown array backend: {array_backend}")


@pytest.mark.parametrize("array_backend", ["numpy"])
def test_array_backend(reference, array_backend):
    message = "`array_backend` is required when using a custom `create_array` callback"
    with pytest.raises(ValueError, match=message):
        metatensor.io.load_custom_array(
            DATA_PATH, create_test_array, array_backend=None
        )

    tensor = metatensor.io.load_custom_array(
        DATA_PATH, create_test_array, array_backend=array_backend
    )

    keys = tensor.keys
    check_array_origin(keys, "python.test-origin")

    check_array_type(keys.values, array_backend)
    assert keys.values.dtype == np.int32
    assert keys.device == "cpu"
    assert keys == reference.keys

    block = tensor.block(0)
    check_array_origin(block, "python.test-origin")

    check_array_type(block.values, array_backend)
    assert_equal(block.values, reference.block(0).values)
    assert block.dtype == np.float64
    assert block.device == "cpu"

    check_array_origin(block.samples, "python.test-origin")
    check_array_type(block.samples.values, array_backend)
    assert block.samples == reference.block(0).samples

    gradient = block.gradient(block.gradients_list()[0])
    check_array_origin(gradient, "python.test-origin")

    check_array_type(gradient.values, array_backend)
    check_array_type(gradient.samples.values, array_backend)

    # operations creating new labels keep the same array backend
    renamed = keys.rename(keys.names[0], "new")
    check_array_type(renamed.values, array_backend)

    appended = keys.append("new", array_to_backend(np.arange(len(keys)), array_backend))
    check_array_type(appended.values, array_backend)
    assert_equal(
        array_to_backend(appended.values[:, :-1], "numpy"), reference.keys.values
    )

    union = keys.union(keys)
    check_array_type(union.values, array_backend)

    # operations creating new TensorMap keep the same array backend
    tensor = tensor.components_to_properties("o3_mu")
    check_array_type(tensor.keys.values, array_backend)
    check_array_type(tensor.block(0).properties.values, array_backend)


def test_invalid_array_backend():
    message = "invalid array_backend 'foo', expected one of 'numpy', 'torch'"
    with pytest.raises(ValueError, match=message):
        metatensor.io.load_custom_array(
            DATA_PATH, create_test_array, array_backend="foo"
        )

    with pytest.raises(ValueError, match=message):
        metatensor.io.load_labels_custom_array(
            os.path.join(os.path.dirname(DATA_PATH), "keys.mts"),
            create_test_array,
            array_backend="foo",
        )


def test_python_arrays_are_returned_as_is():
    # data allocated by Python is returned as-is, regardless of the array_backend
    tensor = metatensor.load(DATA_PATH)
    tensor = metatensor.TensorMap.unsafe_from_ptr(
        tensor.release(), array_backend="torch"
    )

    assert type(tensor.block(0).values) is np.ndarray


def test_to_arrays_deprecated():
    tensor = metatensor.load(DATA_PATH)

    for name, data in [
        ("TensorMap", tensor),
        ("TensorBlock", tensor.block(0)),
        ("Labels", tensor.keys),
    ]:
        message = (
            f"the `arrays` parameter of `{name}.to` is deprecated, "
            "please use `array_backend` instead"
        )
        with pytest.warns(DeprecationWarning, match=message):
            moved = data.to(arrays="torch")

        values = moved.block(0).values if name == "TensorMap" else moved.values
        assert isinstance(values, torch.Tensor)

        message = f"can not give both `arrays` and `array_backend` in `{name}.to`"
        with pytest.raises(ValueError, match=message):
            data.to(arrays="torch", array_backend="torch")
