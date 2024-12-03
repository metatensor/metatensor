import os
import warnings
import zipfile
from pathlib import Path
from typing import Union

import numpy as np

import torch
from metatensor.torch import load_block_buffer
from metatensor.torch import save_buffer as metatensor_torch_save_buffer

from . import (
    NeighborListOptions,
    System,
)


def save(path: Union[str, Path], data: System) -> None:
    """Save a System object to a file.

    The provided System must contain float64 data and be on the CPU device.

    :param path: The path to save the System object to.
    :param data: The System object to save.
    """

    if _is_system(data):
        _save_system(path, data)
    else:
        raise ValueError("`data` must be a System object.")


def load(path: Union[str, Path]) -> System:
    """Load a System object from a file.

    The loaded System object will be on the CPU device and contain float64 data.

    :param path: The path to load the System object from.
    :return: The System object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if _is_system_npz(path):
        return _load_system(path)
    else:
        raise ValueError(f"File does not contain a valid System object: {path}")


def _is_system_npz(path: Union[str, Path]) -> bool:
    zipf = zipfile.ZipFile(path, "r")
    all_zip_files = zipf.namelist()
    required_files = ["positions.npy", "cell.npy", "types.npy", "pbc.npy"]
    for file in required_files:
        if file not in all_zip_files:
            return False
    return True


def _load_system(path: Union[str, Path]) -> System:
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
        for nl_idx in range(len(zipf.namelist())):
            if f"neighbor_lists/{nl_idx}/options.json" in zipf.namelist():
                nl_options = NeighborListOptions.from_json(
                    zipf.read(f"neighbor_lists/{nl_idx}/options.json")
                )
                neighbor_list_options_list.append(nl_options)
                numpy_buffer = np.frombuffer(
                    zipf.read(f"neighbor_lists/{nl_idx}/data.mta"), dtype=np.uint8
                )
                tensor_buffer = torch.from_numpy(
                    numpy_buffer
                ).clone()  # make it writable
                neighbor_list = load_block_buffer(tensor_buffer)
                neighbor_lists.append(neighbor_list)

        extra_data_dict = {}
        for data_name in zipf.namelist():
            if "extra_data/" in data_name:
                key = os.path.basename(data_name).replace(".mta", "")
                numpy_buffer = np.frombuffer(zipf.read(data_name), dtype=np.uint8)
                tensor_buffer = torch.from_numpy(
                    numpy_buffer
                ).clone()  # make it writable
                extra_data_dict[key] = load_block_buffer(tensor_buffer)

    system = System(
        positions=positions,
        cell=cell,
        types=types,
        pbc=pbc,
    )

    for nl_options, nl in zip(neighbor_list_options_list, neighbor_lists):
        system.add_neighbor_list(nl_options, nl)

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
            zipf.writestr(f"neighbor_lists/{nl_idx}/options.json", nl_options.to_json())
            nl = system.get_neighbor_list(nl_options)
            tensor_buffer = metatensor_torch_save_buffer(nl)
            zipf.writestr(
                f"neighbor_lists/{nl_idx}/data.mta", tensor_buffer.numpy().tobytes()
            )
        for key in system.known_data():
            data = system.get_data(key)
            tensor_buffer = metatensor_torch_save_buffer(data)
            zipf.writestr(f"extra_data/{key}.mta", tensor_buffer.numpy().tobytes())

        for tensor_name in ["positions", "cell", "types", "pbc"]:
            tensor = getattr(system, tensor_name)
            numpy_array = tensor.numpy()
            with zipf.open(f"{tensor_name}.npy", "w") as fd:
                np.save(fd, numpy_array)
