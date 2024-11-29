import io
import os
import warnings
import zipfile
from pathlib import Path
from typing import Union

import numpy as np
import torch

import metatensor.torch

from . import NeighborListOptions, System


def save(file: Union[str, Path, io.BytesIO], system: System) -> None:
    """Save a System object to a file.

    The provided System must contain float64 data and be on the CPU device.

    The saved file will be a zip archive containing the following files:

    - ``types.npy``, containing the atomic types in numpy's NPY format;
    - ``positions.npy``, containing the systems' positions in numpy's NPY format;
    - ``cell.npy``, containing the systems' cell in numpy's NPY format;
    - ``pbc.npy``, containing the periodic boundary conditions in numpy's NPY format;

    For each neighbor list in the System object, the following files will be saved
    (where ``{nl_idx}`` is the index of the neighbor list):

    - ``pairs/{nl_idx}/options.json``: the ``NeighborListOptions`` object
      converted to a JSON string.
    - ``pairs/{nl_idx}/data.mts``: the neighbor list ``TensorBlock`` object

    For each extra data in the System object, the following file will be saved (where
    ``{name}`` is the name of the extra data):

    - ``data/{name}.mts``: The extra data ``TensorMap``

    :param file: The path (or file-like object) to save the System to.
    :param system: The System object to save.
    """

    if isinstance(file, (str, Path)):
        if not file.endswith(".mta"):
            raise ValueError("The provided path must have the `.mta` extension.")

    if _is_system(system):
        _save_system(file, system)
    else:
        raise ValueError("`system` must be a System object.")


def load_system(file: Union[str, Path, io.BytesIO]) -> System:
    """Load a System object from a file.

    The loaded System object will be on the CPU device and contain float64 data.

    :param file: The path (or file-like object) to load the System object from.
    :return: The System object.
    """
    if isinstance(file, (str, Path)):
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
        if not file.endswith(".mta"):
            raise ValueError("The provided path must have the `.mta` extension.")

    if _is_system_mta(file):
        return _load_system(file)
    else:
        raise ValueError(f"File does not contain a valid System object: {file}")


def _is_system_mta(path: Union[str, Path, io.BytesIO]) -> bool:
    zipf = zipfile.ZipFile(path, "r")
    all_zip_files = zipf.namelist()
    required_files = ["positions.npy", "cell.npy", "types.npy", "pbc.npy"]
    for file in required_files:
        if file not in all_zip_files:
            return False
    return True


def _load_system(path: Union[str, Path, io.BytesIO]) -> System:
    # we filter a warning related to the fact that numpy arrays from buffers
    # are not writable, while torch would like arrays to be writable when
    # converting them to tensors; this is ok because we then clone the tensor
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="The given NumPy array is not writable"
    )

    with zipfile.ZipFile(path, "r") as zipf:
        positions = torch.from_numpy(np.load(zipf.open("positions.npy")))
        cell = torch.from_numpy(np.load(zipf.open("cell.npy")))
        types = torch.from_numpy(np.load(zipf.open("types.npy")))
        pbc = torch.from_numpy(np.load(zipf.open("pbc.npy")))

        neighbor_list_options_list = []
        neighbor_lists = []
        for path in zipf.namelist():
            if path.startswith("pairs/") and path.endswith("/options.json"):
                nl_options = NeighborListOptions(0.0, False, False)
                nl_options._get_method("__setstate__")(zipf.read(path))
                neighbor_list_options_list.append(nl_options)

                data_path = path[:-12] + "data.mts"
                numpy_buffer = np.frombuffer(zipf.read(data_path), dtype=np.uint8)
                tensor_buffer = torch.from_numpy(numpy_buffer)
                neighbor_lists.append(metatensor.torch.load_block_buffer(tensor_buffer))

        extra_data_dict = {}
        for path in zipf.namelist():
            if path.startswith("data/"):
                name = os.path.basename(path).replace(".mts", "")
                numpy_buffer = np.frombuffer(zipf.read(path), dtype=np.uint8)
                tensor_buffer = torch.from_numpy(numpy_buffer)
                extra_data_dict[name] = metatensor.torch.load_buffer(tensor_buffer)

    system = System(
        positions=positions,
        cell=cell,
        types=types,
        pbc=pbc,
    )

    for options, neighbors in zip(neighbor_list_options_list, neighbor_lists):
        system.add_neighbor_list(options, neighbors)

    for key, value in extra_data_dict.items():
        system.add_data(key, value)

    return system


def _is_system(data: torch.ScriptObject) -> bool:
    if not isinstance(data, torch.ScriptObject):
        return False
    try:
        data.positions
        data.cell
        data.types
        data.pbc
        return True
    except AttributeError:
        return False


def _save_system(path: Union[str, Path], system: System) -> None:
    with zipfile.ZipFile(path, "w") as zipf:
        for nl_idx, nl_options in enumerate(system.known_neighbor_lists()):
            zipf.writestr(f"pairs/{nl_idx}/options.json", nl_options.__getstate__()[0])
            nl = system.get_neighbor_list(nl_options)
            tensor_buffer = metatensor.torch.save_buffer(nl)
            zipf.writestr(f"pairs/{nl_idx}/data.mts", tensor_buffer.numpy().tobytes())

        for key in system.known_data():
            data = system.get_data(key)
            tensor_buffer = metatensor.torch.save_buffer(data)
            zipf.writestr(f"data/{key}.mts", tensor_buffer.numpy().tobytes())

        for tensor_name in ["positions", "cell", "types", "pbc"]:
            tensor = getattr(system, tensor_name)
            numpy_array = tensor.numpy()
            with zipf.open(f"{tensor_name}.npy", "w") as fd:
                np.save(fd, numpy_array)
