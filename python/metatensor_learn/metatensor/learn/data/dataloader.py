"""
Module containing the DataLoader.
"""

import torch

from .collate import group_and_join


class DataLoader(torch.utils.data.DataLoader):
    """
    Class for loading data from an :py:class:`torch.utils.data.Dataset` object with a
    default collate function (:py:func:`metatensor.learn.data.group_and_join`) that
    supports :py:class:`torch.Tensor`, :py:class:`metatensor.torch.atomistic.System`,
    :py:class:`metatensor.TensorMap`, and :py:class:`metatensor.torch.TensorMap`.

    The dataset wil typically be :py:class:`metatensor.learn.Dataset` or
    :py:class:`metatensor.learn.IndexedDataset`.

    Any argument as accepted by the torch :py:class:`torch.utils.data.DataLoader` parent
    class is supported.
    """

    def __init__(self, dataset, collate_fn=group_and_join, **kwargs):
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
