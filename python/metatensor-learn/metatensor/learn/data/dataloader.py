"""
Module containing the DataLoader.
"""

from typing import Callable

import torch

from .collate import group_and_join


class DataLoader(torch.utils.data.DataLoader):
    """
    Class for loading data from an `:py:class:`torch.utils.data.Dataset` object
    with a default collate function that supports :py:class:`torch.Tensor`,
    :py:class:`atomistic.Systems`, or :py:class:`TensorMap`.

    Any argument as accepted by the torch
    :py:class:`torch.utils.data.DataLoader` parent class is supported. Please
    refer to
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        collate_fn: Callable = group_and_join,
        **kwargs,
    ):
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
