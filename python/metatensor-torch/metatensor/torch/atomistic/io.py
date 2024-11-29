import io
import os
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
    Neighbor list requestors will not be saved.

    :param path: The path to save the System object to.
    :param data: The System object to save.
    """

    if not isinstance(data, torch.ScriptObject):
        raise ValueError("`data` must be a scripted object.")

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
    try:
        data = np.load(path, allow_pickle=True)
        data["positions"]
        data["cell"]
        data["types"]
        data["neighbor_lists"]
        return True
    except Exception:
        return False


def _load_system(path: Union[str, Path]) -> System:
    data = np.load(path, allow_pickle=True)
    positions = torch.from_numpy(data["positions"])
    cell = torch.from_numpy(data["cell"])
    types = torch.from_numpy(data["types"])
    pbc = torch.from_numpy(data["pbc"])

    neighbor_list_dict = np.load(io.BytesIO(data["neighbor_lists"]), allow_pickle=True)
    neighbor_list_dict = {
        key: torch.from_numpy(value) for key, value in neighbor_list_dict.items()
    }

    extra_data_dict = np.load(io.BytesIO(data["extra_data"]), allow_pickle=True)
    extra_data_dict = {
        key: torch.from_numpy(value) for key, value in extra_data_dict.items()
    }

    system = System(
        positions=positions,
        cell=cell,
        types=types,
        pbc=pbc,
    )

    for nl_key, nl in neighbor_list_dict.items():
        cutoff, full_list, strict = nl_key.split("-")
        cutoff = float(cutoff)
        full_list = full_list == "True"
        strict = strict == "True"
        nl_options = NeighborListOptions(
            cutoff=cutoff, full_list=full_list, strict=strict
        )
        nl = load_block_buffer(nl)
        system.add_neighbor_list(nl_options, nl)

    for key, value in extra_data_dict.items():
        value = load_block_buffer(value)
        system.add_data(key, value)

    return system


def _is_system(data: torch.ScriptObject) -> bool:
    try:
        data.positions
        data.cell
        data.types
        data.pbc
        return True
    except AttributeError:
        return False


def _save_system(path: Union[str, Path], system: System) -> None:
    neighbor_lists_buffer = io.BytesIO()
    neighbor_list_dict = {}
    for nl_options in system.known_neighbor_lists():
        nl = system.get_neighbor_list(nl_options)
        nl_key = f"{nl_options.cutoff}-{nl_options.full_list}-{nl_options.strict}"
        tensor_buffer = metatensor_torch_save_buffer(nl)
        numpy_buffer = tensor_buffer.numpy()
        neighbor_list_dict[nl_key] = numpy_buffer
    np.savez(neighbor_lists_buffer, **neighbor_list_dict)

    extra_data_buffer = io.BytesIO()
    extra_data_dict = {}
    for key in system.known_data():
        data = system.get_data(key)
        tensor_buffer = metatensor_torch_save_buffer(data)
        numpy_buffer = tensor_buffer.numpy()
        extra_data_dict[key] = numpy_buffer
    np.savez(extra_data_buffer, **extra_data_dict)

    system_dict = {
        "positions": system.positions.numpy(),
        "cell": system.cell.numpy(),
        "types": system.types.numpy(),
        "pbc": system.pbc.numpy(),
        "neighbor_lists": neighbor_lists_buffer.getvalue(),
        "extra_data": extra_data_buffer.getvalue(),
    }

    np.savez(path, **system_dict)
