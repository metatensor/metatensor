import os

import torch

import metatensor.core
from metatensor.torch import Labels, TensorBlock, TensorMap


ROOT = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-operations",
        "tests",
        "data",
    )
)


def load_data(data: str):
    # we need to load from metatensor-core and then convert to the torch types since
    # the files are using DEFLATE compression, while metatensor only supports STORED
    tensor = metatensor.core.load(os.path.join(ROOT, data), use_numpy=True)

    blocks = []
    for block in tensor:
        blocks.append(_block_to_torch(block))

    return TensorMap(_labels_to_torch(tensor.keys), blocks)


def _labels_to_torch(labels):
    return Labels(labels.names, torch.tensor(labels.values))


def _block_to_torch(block):
    new_block = TensorBlock(
        torch.tensor(block.values),
        _labels_to_torch(block.samples),
        [_labels_to_torch(c) for c in block.components],
        _labels_to_torch(block.properties),
    )

    for parameter, gradient in block.gradients():
        new_block.add_gradient(parameter, _block_to_torch(gradient))

    return new_block
