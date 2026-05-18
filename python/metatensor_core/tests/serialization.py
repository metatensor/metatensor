import io
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

import metatensor as mts
from metatensor import Labels, MetatensorError, TensorBlock, TensorMap

from . import _tests_utils


PICKLE_PROTOCOLS = (4, 5)


@pytest.fixture
def tensor():
    return _tests_utils.tensor()


@pytest.fixture
def block(tensor):
    return tensor.block(1)


@pytest.fixture
def labels():
    return _tests_utils.tensor().keys


@pytest.fixture
def tensor_zero_len_block():
    return _tests_utils.tensor_zero_len_block()


@pytest.mark.parametrize("use_numpy", (True, False))
@pytest.mark.parametrize("memory_buffer", (True, False))
@pytest.mark.parametrize("standalone_fn", (True, False))
def test_load(use_numpy, memory_buffer, standalone_fn):
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "data.mts",
    )

    if memory_buffer:
        with open(path, "rb") as fd:
            buffer = fd.read()

        assert isinstance(buffer, bytes)
    else:
        file = path

    if standalone_fn:
        if memory_buffer:
            tensor = mts.io.load_buffer(buffer, use_numpy=use_numpy)
        else:
            tensor = mts.load(file, use_numpy=use_numpy)
    else:
        if memory_buffer:
            tensor = TensorMap.load_buffer(buffer, use_numpy=use_numpy)
        else:
            tensor = TensorMap.load(file, use_numpy=use_numpy)

    assert isinstance(tensor, TensorMap)
    assert tensor.keys.names == [
        "o3_lambda",
        "o3_sigma",
        "center_type",
        "neighbor_type",
    ]
    assert len(tensor.keys) == 27

    block = tensor.block(o3_lambda=2, center_type=6, neighbor_type=1)
    assert block.samples.names == ["system", "atom"]
    assert block.values.shape == (9, 5, 3)

    gradient = block.gradient("positions")
    assert gradient.samples.names == ["sample", "system", "atom"]
    assert gradient.values.shape == (59, 3, 5, 3)


def _data_mts_path():
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "data.mts",
    )


def _block_mts_path():
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "block.mts",
    )


@pytest.mark.parametrize("standalone_fn", (True, False))
def test_load_mmap_tensor(standalone_fn):
    path = _data_mts_path()

    if standalone_fn:
        tensor = mts.load_mmap(path)
    else:
        tensor = TensorMap.load(path)  # canonical reference

        # The mmap loader must produce a TensorMap with identical structure
        # and equal values to the canonical streaming loader.
        mmap_tensor = mts.io.load_mmap(path)
        assert isinstance(mmap_tensor, TensorMap)
        assert mmap_tensor.keys.names == tensor.keys.names
        assert len(mmap_tensor.keys) == len(tensor.keys)
        for ref, got in zip(tensor.blocks(), mmap_tensor.blocks(), strict=True):
            assert got.values.shape == ref.values.shape
            np.testing.assert_array_equal(
                np.asarray(got.values), np.asarray(ref.values)
            )
        return

    assert isinstance(tensor, TensorMap)
    assert tensor.keys.names == [
        "o3_lambda",
        "o3_sigma",
        "center_type",
        "neighbor_type",
    ]
    assert len(tensor.keys) == 27

    block = tensor.block(o3_lambda=2, center_type=6, neighbor_type=1)
    assert block.samples.names == ["system", "atom"]
    assert block.values.shape == (9, 5, 3)
    assert not block.values.flags.writeable, "mmap arrays should be read-only"

    gradient = block.gradient("positions")
    assert gradient.samples.names == ["sample", "system", "atom"]
    assert gradient.values.shape == (59, 3, 5, 3)


def test_load_mmap_block():
    path = _block_mts_path()

    block = mts.io.load_block_mmap(path)
    assert isinstance(block, TensorBlock)
    assert block.values.shape[0] > 0
    assert not block.values.flags.writeable


def test_load_mmap_pathlib():
    # pathlib.Path inputs must work alongside str inputs.
    path = Path(_data_mts_path())
    tensor = mts.io.load_mmap(path)
    assert isinstance(tensor, TensorMap)


def test_load_partial_select_all():
    # No filters: must equal canonical load.
    path = _data_mts_path()
    ref = mts.load(path)
    got = mts.load_partial(path)
    assert isinstance(got, TensorMap)
    assert got.keys.names == ref.keys.names
    assert len(got.keys) == len(ref.keys)
    for ref_block, got_block in zip(ref.blocks(), got.blocks(), strict=True):
        assert got_block.values.shape == ref_block.values.shape
        np.testing.assert_array_equal(
            np.asarray(got_block.values), np.asarray(ref_block.values)
        )


def test_load_partial_filter_keys():
    path = _data_mts_path()
    keys_filter = Labels(
        names=["o3_lambda", "center_type"],
        values=np.array([[2, 6]], dtype=np.int32),
    )
    got = mts.load_partial(path, keys=keys_filter)
    assert isinstance(got, TensorMap)
    # the matching keys filter selects 4 (o3_lambda=2, center_type=6,
    # neighbor_type in {1,6,8}, o3_sigma=1)
    assert len(got.keys) >= 1
    for entry in got.keys:
        assert entry["o3_lambda"] == 2
        assert entry["center_type"] == 6


def test_load_partial_filter_samples():
    path = _data_mts_path()
    # pick a sample selector that's present in the data
    ref = mts.load(path)
    ref_block = ref.block(o3_lambda=2, center_type=6, neighbor_type=1)
    first_sample = np.array(ref_block.samples.values[:1], dtype=np.int32)
    samples_filter = Labels(names=ref_block.samples.names, values=first_sample)

    got = mts.load_partial(path, samples=samples_filter)
    got_block = got.block(o3_lambda=2, center_type=6, neighbor_type=1)
    assert got_block.samples.values.shape[0] == 1
    np.testing.assert_array_equal(np.asarray(got_block.samples.values), first_sample)


def test_load_partial_filter_properties():
    """
    Property selection drives the per-row scratch-buffer branch of
    gather_selected_data (one pread per kept row into row_buf, then
    column-subset scatter into dst). Verify the kept columns match the
    canonical load byte-for-byte, in the requested order.
    """
    path = _data_mts_path()
    ref = mts.load(path)
    ref_block = ref.block(o3_lambda=2, center_type=6, neighbor_type=1)
    assert ref_block.properties.values.shape[0] >= 2, (
        "test assumes the reference block has at least two properties"
    )

    # Keep the last property first, then the first; tests both subset
    # selection and column reordering through the scratch buffer.
    kept = ref_block.properties.values[[-1, 0], :].astype(np.int32, copy=True)
    properties_filter = Labels(names=ref_block.properties.names, values=kept)

    got = mts.load_partial(path, properties=properties_filter)
    got_block = got.block(o3_lambda=2, center_type=6, neighbor_type=1)
    assert got_block.properties.values.shape[0] == 2
    np.testing.assert_array_equal(np.asarray(got_block.properties.values), kept)

    # Every selected column must match the canonical block at the
    # same property index, in the requested order.
    ref_values = np.asarray(ref_block.values)
    got_values = np.asarray(got_block.values)
    ref_cols = [
        int(np.where((ref_block.properties.values == row).all(axis=1))[0][0])
        for row in kept
    ]
    np.testing.assert_array_equal(got_values, ref_values[..., ref_cols])


def test_load_partial_nested_gradients(tmpdir):
    block = TensorBlock(
        values=np.arange(9, dtype=np.float64).reshape(3, 3),
        samples=Labels.range("s", 3),
        components=[],
        properties=Labels.range("p", 3),
    )

    gradient = TensorBlock(
        values=np.arange(9, 18, dtype=np.float64).reshape(3, 3),
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels.range("p", 3),
    )

    grad_grad = TensorBlock(
        values=np.arange(45, dtype=np.float64).reshape(3, 5, 3),
        samples=Labels.range("sample", 3),
        components=[Labels.range("c", 5)],
        properties=Labels.range("p", 3),
    )

    block_values = block.values.copy()
    gradient_values = gradient.values.copy()
    grad_grad_values = grad_grad.values.copy()

    gradient.add_gradient("grad-of-grad", grad_grad)
    block.add_gradient("grad", gradient)
    tensor = TensorMap(Labels.single(), [block])

    samples_filter = Labels(
        names=["s"],
        values=np.array([[0], [2]], dtype=np.int32),
    )

    tmpfile = "partial-grad-grad-test.mts"
    with tmpdir.as_cwd():
        mts.save(tmpfile, tensor)
        loaded = mts.load_partial(tmpfile, samples=samples_filter)

    loaded_block = loaded.block(0)
    loaded_gradient = loaded_block.gradient("grad")
    loaded_grad_grad = loaded_gradient.gradient("grad-of-grad")

    np.testing.assert_array_equal(loaded_block.values, block_values[[0, 2], :])
    np.testing.assert_array_equal(loaded_gradient.values, gradient_values[[0, 2], :])
    np.testing.assert_array_equal(
        loaded_gradient.samples.values,
        np.array([[0], [1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        loaded_grad_grad.values,
        grad_grad_values[[0, 2], :, :],
    )
    np.testing.assert_array_equal(
        loaded_grad_grad.samples.values,
        np.array([[0], [1]], dtype=np.int32),
    )


def test_load_block_partial_round_trip():
    path = _block_mts_path()
    block = mts.io.load_block_partial(path)
    assert isinstance(block, TensorBlock)
    assert block.values.shape[0] > 0


def test_load_mmap_gds_values_equal():
    """
    Load a TensorMap via NVIDIA GPU Direct Storage (cuFile) and verify
    every value array equals the canonical CPU load, byte-for-byte.

    Skipped on systems without cupy + kvikio. When kvikio is in compat
    mode (no nvidia-fs kernel module or non-GDS-capable filesystem),
    cuFile still reads via POSIX but with a host bounce buffer; the
    end result is the same GPU array, so we test both modes via the
    same assertion.
    """
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("kvikio")
    from metatensor.io._mmap_gds import is_using_real_gds, load_mmap_gds

    path = _data_mts_path()
    ref = mts.load(path)
    gpu = load_mmap_gds(path)

    print(f"GDS mode: {'direct DMA' if is_using_real_gds() else 'compat (POSIX)'}")
    assert isinstance(gpu, TensorMap)
    assert gpu.keys.names == ref.keys.names
    assert len(gpu.keys) == len(ref.keys)

    for ref_block, gpu_block in zip(ref.blocks(), gpu.blocks(), strict=True):
        ref_np = np.asarray(ref_block.values)
        gpu_np = cupy.asnumpy(gpu_block.values)
        assert gpu_np.shape == ref_np.shape
        assert gpu_np.dtype == ref_np.dtype
        np.testing.assert_array_equal(gpu_np, ref_np)


def test_load_partial_mmap_gds_round_trip():
    """
    Multi-region GDS partial load: select half the blocks and one
    sample row per surviving block; verify the GPU result equals the
    canonical mts.load_partial result, byte-for-byte.
    """
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("kvikio")
    from metatensor.io._mmap_gds import load_partial_mmap_gds

    path = _data_mts_path()

    # No filter at all: must equal full mts.load.
    ref_full = mts.load(path)
    gpu_full = load_partial_mmap_gds(path)
    assert len(gpu_full.keys) == len(ref_full.keys)
    for ref_block, gpu_block in zip(ref_full.blocks(), gpu_full.blocks(), strict=True):
        ref_np = np.asarray(ref_block.values)
        gpu_np = cupy.asnumpy(gpu_block.values)
        np.testing.assert_array_equal(gpu_np, ref_np)

    # Key + sample filter.
    keys_filter = Labels(
        names=["o3_lambda", "center_type"],
        values=np.array([[2, 6]], dtype=np.int32),
    )
    samples_filter = Labels(
        names=["system"],
        values=np.array([[0]], dtype=np.int32),
    )
    ref_filtered = mts.load_partial(path, keys=keys_filter, samples=samples_filter)
    gpu_filtered = load_partial_mmap_gds(path, keys=keys_filter, samples=samples_filter)
    assert len(gpu_filtered.keys) == len(ref_filtered.keys)
    for ref_block, gpu_block in zip(
        ref_filtered.blocks(), gpu_filtered.blocks(), strict=True
    ):
        ref_np = np.asarray(ref_block.values)
        gpu_np = cupy.asnumpy(gpu_block.values)
        np.testing.assert_array_equal(gpu_np, ref_np)


def test_load_block_mmap_gds_values_equal():
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("kvikio")
    from metatensor.io._mmap_gds import load_block_mmap_gds

    path = _block_mts_path()
    ref = mts.io.load_block(path)
    gpu = load_block_mmap_gds(path)
    assert isinstance(gpu, TensorBlock)
    ref_np = np.asarray(ref.values)
    gpu_np = cupy.asnumpy(gpu.values)
    np.testing.assert_array_equal(gpu_np, ref_np)


def test_load_mmap_values_are_views():
    # mmap-loaded arrays must be views into a memory-mapped buffer,
    # not freshly-allocated copies. We assert: (a) values are read-only
    # and (b) the .base chain leads to *some* non-None buffer (numpy
    # arrays whose data is a fresh malloc would have base=None).
    path = _data_mts_path()
    tensor = mts.io.load_mmap(path)
    for block in tensor.blocks():
        arr = np.asarray(block.values)
        assert not arr.flags.writeable, "mmap arrays should be read-only"
        # walk to the lowest non-None base
        base = arr.base
        while base is not None and getattr(base, "base", None) is not None:
            base = base.base
        assert base is not None, (
            "expected the array to be a view into an mmap, not an "
            "independently-allocated buffer (base is None)"
        )


@pytest.mark.parametrize("use_numpy", (True, False))
def test_load_deflate(use_numpy):
    # This file was saved using DEFLATE to compress the different ZIP archive members
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "metatensor_operations",
        "tests",
        "data",
        "qm7-power-spectrum.mts",
    )

    tensor = mts.load(path, use_numpy=use_numpy)

    assert isinstance(tensor, TensorMap)
    assert tensor.keys.names == [
        "center_type",
        "neighbor_1_type",
        "neighbor_2_type",
    ]
    assert len(tensor.keys) == 17


# using tmpdir as pytest-built-in fixture
# https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html#the-tmpdir-and-tmpdir-factory-fixtures
@pytest.mark.parametrize("use_numpy", (True, False))
@pytest.mark.parametrize("memory_buffer", (True, False))
@pytest.mark.parametrize("standalone_fn", (True, False))
def test_save(use_numpy, memory_buffer, standalone_fn, tmpdir, tensor):
    """Check that as saved file loads fine with numpy."""
    with tmpdir.as_cwd():
        if memory_buffer:
            if standalone_fn:
                buffer = mts.io.save_buffer(tensor, use_numpy=use_numpy)
            else:
                buffer = tensor.save_buffer(use_numpy=use_numpy)

            size = len(buffer)
            file = io.BytesIO(buffer)

        else:
            file = "serialize-test.mts"
            if standalone_fn:
                mts.save(file, tensor, use_numpy=use_numpy)
            else:
                tensor.save(file, use_numpy=use_numpy)

            size = os.path.getsize(file)

        assert size == 8718
        data = np.load(file)

    assert len(data.keys()) == 29

    assert _mts_labels(data["keys"]) == tensor.keys
    for i, block in enumerate(tensor.blocks()):
        prefix = f"blocks/{i}"

        np.testing.assert_equal(data[f"{prefix}/values"], block.values)
        assert _mts_labels(data[f"{prefix}/samples"]) == block.samples
        assert _mts_labels(data[f"{prefix}/components/0"]) == block.components[0]
        assert _mts_labels(data[f"{prefix}/properties"]) == block.properties

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            prefix = f"blocks/{i}/gradients/{parameter}"

            np.testing.assert_equal(data[f"{prefix}/values"], gradient.values)
            assert _mts_labels(data[f"{prefix}/samples"]) == gradient.samples
            assert _mts_labels(data[f"{prefix}/components/0"]) == gradient.components[0]


def test_save_buffer(tensor):
    # check that we can save/load without going through a file
    buffer = tensor.save_buffer()
    assert isinstance(buffer, memoryview)
    loaded = TensorMap.load_buffer(buffer)

    assert loaded.keys == tensor.keys


@pytest.mark.parametrize("use_numpy_save", (True, False))
@pytest.mark.parametrize("use_numpy_load", (True, False))
def test_save_load_zero_length_block(
    use_numpy_save, use_numpy_load, tmpdir, tensor_zero_len_block
):
    """
    Tests that attempting to save and load a TensorMap with a zero-length axis block
    does not raise an error, when using combinations of use_numpy for save and
    load
    """
    file = "serialize-test-zero-len-block.mts"

    with tmpdir.as_cwd():
        mts.save(file, tensor_zero_len_block, use_numpy=use_numpy_save)
        mts.load(file, use_numpy=use_numpy_load)


def test_save_warning_errors(tmpdir, tensor):
    # does not have .mts ending and causes warning
    tmpfile = "serialize-test"

    with pytest.warns() as record:
        with tmpdir.as_cwd():
            mts.save(tmpfile, tensor)

    expected = f"adding '.mts' extension, the file will be saved at '{tmpfile}.mts'"
    assert str(record[0].message) == expected

    tmpfile = "serialize-test.mts"

    message = (
        "`data` must be one of 'Labels', 'TensorBlock' or 'TensorMap', "
        "not <class 'numpy.ndarray'>"
    )
    with pytest.raises(TypeError, match=message):
        with tmpdir.as_cwd():
            mts.save(tmpfile, tensor.block(0).values)


def test_save_pathlib(tmpdir, tensor):
    # does not have .mts ending and causes warning
    tmpfile = Path("serialize-test")

    expected = f"adding '.mts' extension, the file will be saved at '{tmpfile}.mts'"
    with tmpdir.as_cwd():
        with pytest.warns(UserWarning, match=expected):
            mts.save(tmpfile, tensor)

    tmpfile = "serialize-test.mts"

    message = (
        "`data` must be one of 'Labels', 'TensorBlock' or 'TensorMap', "
        "not <class 'numpy.ndarray'>"
    )
    with pytest.raises(TypeError, match=message):
        with tmpdir.as_cwd():
            mts.save(tmpfile, tensor.block(0).values)


@pytest.mark.parametrize("protocol", PICKLE_PROTOCOLS)
def test_pickle(protocol, tmpdir, tensor):
    """
    Checks that pickling and unpickling a tensor map results in the same tensor map
    """

    tmpfile = "serialize-test.pickle"

    with tmpdir.as_cwd():
        with open(tmpfile, "wb") as f:
            pickle.dump(tensor, f, protocol=protocol)

        with open(tmpfile, "rb") as f:
            tensor_loaded = pickle.load(f)

    np.testing.assert_equal(tensor.keys, tensor_loaded.keys)
    for key, block in tensor_loaded.items():
        ref_block = tensor.block(key)

        assert isinstance(block.values, np.ndarray)
        np.testing.assert_equal(block.values, ref_block.values)

        assert block.samples == ref_block.samples
        assert block.components == ref_block.components
        assert block.properties == ref_block.properties

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            ref_gradient = ref_block.gradient(parameter)

            np.testing.assert_equal(gradient.values, ref_gradient.values)

            assert gradient.samples == ref_gradient.samples
            assert gradient.components == ref_gradient.components


@pytest.mark.parametrize("use_numpy", (True, False))
def test_nested_gradients(tmpdir, use_numpy):
    block = TensorBlock(
        values=np.random.rand(3, 3),
        samples=Labels.range("s", 3),
        components=[],
        properties=Labels.range("p", 3),
    )

    grad = TensorBlock(
        values=np.random.rand(3, 3),
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels.range("p", 3),
    )

    grad_grad = TensorBlock(
        values=np.random.rand(3, 5, 3),
        samples=Labels.range("sample", 3),
        components=[Labels.range("c", 5)],
        properties=Labels.range("p", 3),
    )

    grad.add_gradient("grad-of-grad", grad_grad)
    block.add_gradient("grad", grad)
    tensor = TensorMap(Labels.single(), [block])

    tmpfile = "grad-grad-test.mts"

    with tmpdir.as_cwd():
        mts.save(tmpfile, tensor, use_numpy=use_numpy)

        # load back with numpy
        data = np.load(tmpfile)

        # load back with metatensor
        loaded = mts.load(tmpfile)

    assert _mts_labels(data["keys"]) == tensor.keys
    assert _mts_labels(data["keys"]) == loaded.keys

    for i, (key, block) in enumerate(tensor.items()):
        loaded_block = loaded.block(key)

        prefix = f"blocks/{i}"
        np.testing.assert_equal(data[f"{prefix}/values"], block.values)
        np.testing.assert_equal(block.values, loaded_block.values)

        assert _mts_labels(data[f"{prefix}/samples"]) == block.samples
        assert block.samples == loaded_block.samples

        assert _mts_labels(data[f"{prefix}/properties"]) == block.properties
        assert block.properties == loaded_block.properties

        assert block.gradients_list() == loaded_block.gradients_list()

        for parameter, gradient in block.gradients():
            loaded_gradient = loaded_block.gradient(parameter)
            grad_prefix = f"{prefix}/gradients/{parameter}"

            np.testing.assert_equal(data[f"{grad_prefix}/values"], gradient.values)
            np.testing.assert_equal(gradient.values, loaded_gradient.values)

            assert _mts_labels(data[f"{grad_prefix}/samples"]) == gradient.samples
            assert gradient.samples == loaded_gradient.samples

            assert gradient.components == loaded_gradient.components
            assert gradient.properties == loaded_gradient.properties

            assert gradient.gradients_list() == loaded_gradient.gradients_list()

            for parameter, grad_grad in gradient.gradients():
                loaded = loaded_gradient.gradient(parameter)
                grad_grad_prefix = f"{grad_prefix}/gradients/{parameter}"

                np.testing.assert_equal(
                    data[f"{grad_grad_prefix}/values"], grad_grad.values
                )
                np.testing.assert_equal(grad_grad.values, loaded.values)

                assert (
                    _mts_labels(data[f"{grad_grad_prefix}/samples"])
                    == grad_grad.samples
                )

                assert grad_grad.samples == loaded.samples

                assert len(grad_grad.components) == len(loaded.components)
                for a, b in zip(grad_grad.components, loaded.components, strict=True):
                    assert a == b

                assert grad_grad.properties == loaded.properties


def _mts_labels(data):
    names = data.dtype.names
    return Labels(names=names, values=data.view(dtype=np.int32).reshape(-1, len(names)))


@pytest.mark.parametrize("memory_buffer", (True, False))
@pytest.mark.parametrize("standalone_fn", (True, False))
def test_load_labels(memory_buffer, standalone_fn):
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "keys.mts",
    )

    if memory_buffer:
        with open(path, "rb") as fd:
            buffer = fd.read()

        assert isinstance(buffer, bytes)
    else:
        file = path

    if standalone_fn:
        if memory_buffer:
            labels = mts.io.load_labels_buffer(buffer)
        else:
            labels = mts.load_labels(file)
    else:
        if memory_buffer:
            labels = Labels.load_buffer(buffer)
        else:
            labels = Labels.load(file)

    assert isinstance(labels, Labels)
    assert labels.names == [
        "o3_lambda",
        "o3_sigma",
        "center_type",
        "neighbor_type",
    ]
    assert len(labels) == 27


@pytest.mark.parametrize("memory_buffer", (True, False))
@pytest.mark.parametrize("standalone_fn", (True, False))
def test_save_labels(memory_buffer, standalone_fn, tmpdir, labels):
    """Check that as saved file loads fine with numpy."""
    with tmpdir.as_cwd():
        if memory_buffer:
            if standalone_fn:
                buffer = mts.io.save_buffer(labels)
            else:
                buffer = labels.save_buffer()

            file = io.BytesIO(buffer)
        else:
            file = "serialize-test.mts"
            if standalone_fn:
                mts.save(file, labels)
            else:
                labels.save(file)

        data = np.load(file)

    assert _mts_labels(data) == labels


def test_save_labels_buffer(labels):
    # check that we can save/load without going through a file
    buffer = labels.save_buffer()
    assert isinstance(buffer, memoryview)
    loaded = Labels.load_buffer(buffer)

    assert labels == loaded


@pytest.mark.parametrize("protocol", PICKLE_PROTOCOLS)
def test_pickle_labels(protocol, tmpdir, labels):
    """
    Checks that pickling and unpickling Labels results in the same Labels
    """

    tmpfile = "serialize-test.pickle"

    with tmpdir.as_cwd():
        with open(tmpfile, "wb") as f:
            pickle.dump(labels, f, protocol=protocol)

        with open(tmpfile, "rb") as f:
            labels_loaded = pickle.load(f)

    np.testing.assert_equal(labels, labels_loaded)


def test_wrong_load_error():
    data_root = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "metatensor-core", "tests"
    )

    message = (
        "serialization format error: unable to load a TensorMap from '.*', "
        "use `load_labels` to load Labels"
    )
    with pytest.raises(MetatensorError, match=message):
        mts.load(os.path.join(data_root, "keys.mts"))

    message = (
        "serialization format error: unable to load a TensorMap from buffer, "
        "use `load_labels_buffer` to load Labels"
    )
    with pytest.raises(MetatensorError, match=message):
        with open(os.path.join(data_root, "keys.mts"), "rb") as fd:
            buffer = fd.read()

        mts.load(io.BytesIO(buffer))

    message = (
        "serialization format error: unable to load Labels from '.*', "
        "use `load` to load TensorMap: start does not match magic string"
    )
    with pytest.raises(MetatensorError, match=message):
        mts.load_labels(os.path.join(data_root, "data.mts"))

    message = (
        "serialization format error: unable to load Labels from buffer, "
        "use `load_buffer` to load TensorMap: start does not match magic string"
    )
    with pytest.raises(MetatensorError, match=message):
        with open(os.path.join(data_root, "data.mts"), "rb") as fd:
            buffer = fd.read()

        mts.load_labels(io.BytesIO(buffer))


@pytest.mark.parametrize("use_numpy", (True, False))
@pytest.mark.parametrize("memory_buffer", (True, False))
@pytest.mark.parametrize("standalone_fn", (True, False))
def test_load_block(use_numpy, memory_buffer, standalone_fn):
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "block.mts",
    )

    if memory_buffer:
        with open(path, "rb") as fd:
            buffer = fd.read()

        assert isinstance(buffer, bytes)
    else:
        file = path

    if standalone_fn:
        if memory_buffer:
            block = mts.io.load_block_buffer(buffer, use_numpy=use_numpy)
        else:
            block = mts.load_block(file, use_numpy=use_numpy)
    else:
        if memory_buffer:
            block = TensorBlock.load_buffer(buffer, use_numpy=use_numpy)
        else:
            block = TensorBlock.load(file, use_numpy=use_numpy)

    assert isinstance(block, TensorBlock)
    assert block.samples.names == ["system", "atom"]
    assert block.values.shape == (9, 5, 3)

    gradient = block.gradient("positions")
    assert gradient.samples.names == ["sample", "system", "atom"]
    assert gradient.values.shape == (59, 3, 5, 3)


@pytest.mark.parametrize("use_numpy", (True, False))
@pytest.mark.parametrize("memory_buffer", (True, False))
@pytest.mark.parametrize("standalone_fn", (True, False))
def test_save_block(use_numpy, memory_buffer, standalone_fn, tmpdir, block):
    with tmpdir.as_cwd():
        if memory_buffer:
            if standalone_fn:
                buffer = mts.io.save_buffer(block, use_numpy=use_numpy)
            else:
                buffer = block.save_buffer(use_numpy=use_numpy)

            file = io.BytesIO(buffer)

        else:
            file = "serialize-test.mts"
            if standalone_fn:
                mts.save(file, block, use_numpy=use_numpy)
            else:
                block.save(file, use_numpy=use_numpy)

        data = np.load(file)

    assert len(data.keys()) == 7

    np.testing.assert_equal(data["values"], block.values)
    assert _mts_labels(data["samples"]) == block.samples
    assert _mts_labels(data["components/0"]) == block.components[0]
    assert _mts_labels(data["properties"]) == block.properties

    for parameter in block.gradients_list():
        gradient = block.gradient(parameter)
        prefix = f"gradients/{parameter}"

        np.testing.assert_equal(data[f"{prefix}/values"], gradient.values)
        assert _mts_labels(data[f"{prefix}/samples"]) == gradient.samples
        assert _mts_labels(data[f"{prefix}/components/0"]) == gradient.components[0]


def test_save_block_buffer(block):
    # check that we can save/load without going through a file
    buffer = block.save_buffer()
    assert isinstance(buffer, memoryview)
    loaded = TensorBlock.load_buffer(buffer)

    np.testing.assert_equal(loaded.values, block.values)
    assert loaded.samples == block.samples
    assert loaded.components[0] == block.components[0]
    assert loaded.properties == block.properties


@pytest.mark.parametrize("use_numpy", (True, False))
def test_save_load_info(tensor, use_numpy):
    tensor.set_info("test", "value")

    buffer = mts.io.save_buffer(tensor, use_numpy=use_numpy)
    assert len(buffer) == 8853

    data = np.load(io.BytesIO(buffer))

    assert json.loads(data["info.json"].decode("utf8")) == {"test": "value"}

    # check that loading back works both with and without numpy
    for use_numpy_load in (False, True):
        loaded = mts.io.load_buffer(buffer, use_numpy=use_numpy_load)
        assert loaded.get_info("test") == "value"


@pytest.mark.parametrize("use_numpy", (True, False))
@pytest.mark.parametrize(
    "dtype",
    [
        # Standard Numeric
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,  # aka np.half
        np.float32,
        np.float64,
        # Boolean
        np.bool_,
        # Complex
        np.complex64,
        np.complex128,
    ],
)
def test_save_dtypes(tmp_path, dtype, use_numpy):
    data = np.arange(6).reshape(2, 3).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        data += 0.1

    block = TensorBlock(
        values=data,
        samples=Labels(["s"], np.array([[0], [1]], dtype=np.int32)),
        components=[],
        properties=Labels(
            ["p"], np.arange(data.shape[1], dtype=np.int32).reshape(-1, 1)
        ),
    )
    keys = Labels(["key"], np.array([[0]], dtype=np.int32))
    tensor = TensorMap(keys, [block])

    file_path = tmp_path / f"test_dtype_{dtype.__name__}.mts"

    # Python -> C-API -> Rust -> DLPack -> serialization
    mts.save(file_path, tensor, use_numpy=False)

    # Verify full round-trip via both native (use_numpy=False) and numpy path
    loaded_tensor = mts.load(file_path, use_numpy=use_numpy)
    vals_loaded = loaded_tensor.block(0).values

    assert vals_loaded.dtype == dtype
    np.testing.assert_array_equal(vals_loaded, data)

    # Verify with NumPy directly (ZIP archive of .npy files)
    archive = np.load(file_path)
    vals_loaded = archive["blocks/0/values"]
    assert vals_loaded.dtype == dtype
    np.testing.assert_array_equal(vals_loaded, data)


@pytest.mark.parametrize("use_numpy", (True, False))
def test_save_fortran(tmp_path, use_numpy):
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64, order="F")

    block = TensorBlock(
        values=data,
        samples=Labels.range("s", 2),
        components=[],
        properties=Labels.range("p", 3),
    )
    tensor = TensorMap(Labels.single(), [block])

    file_path = tmp_path / "strided_test.mts"

    mts.save(file_path, tensor, use_numpy=use_numpy)

    loaded = mts.load(file_path, use_numpy=True)

    np.testing.assert_array_equal(loaded.block(0).values, data)


@pytest.mark.parametrize("use_numpy", (True, False))
def test_save_strided(tmp_path, use_numpy):
    data = np.arange(20, dtype=np.float64).reshape(4, 5)[:, ::2]

    block = TensorBlock(
        values=data,
        samples=Labels.range("s", 4),
        components=[],
        properties=Labels.range("p", 3),
    )
    tensor = TensorMap(Labels.single(), [block])

    file_path = tmp_path / "strided_test.mts"

    mts.save(file_path, tensor, use_numpy=use_numpy)
    loaded = mts.load(file_path, use_numpy=True)

    np.testing.assert_array_equal(loaded.block(0).values, data)
