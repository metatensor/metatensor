import sys
import torch

from ._c_lib import _load_library


if (sys.version_info.major >= 3) and (sys.version_info.minor >= 8):
    import importlib.metadata

    __version__ = importlib.metadata.version("equistore-torch")

else:
    from pkg_resources import get_distribution

    __version__ = get_distribution("equistore-torch").version


_load_library()

Labels = torch.classes.equistore.Labels
TensorBlock = torch.classes.equistore.TensorBlock
TensorMap = torch.classes.equistore.TensorMap


__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
