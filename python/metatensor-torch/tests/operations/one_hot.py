import io

import torch

import metatensor.torch


def check_operation(one_hot):
    original_labels = metatensor.torch.Labels(
        names=["atom", "species"],
        values=torch.tensor([[0, 6], [1, 1], [2, 1], [3, 1], [4, 6], [5, 1]]),
    )
    possible_labels = metatensor.torch.Labels(
        names=["species"], values=torch.tensor([[1], [6]])
    )
    correct_encoding = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 0],
        ],
        dtype=torch.int32,
    )
    one_hot_encoding = one_hot(original_labels, possible_labels)

    # check output type
    assert isinstance(one_hot_encoding, torch.Tensor)

    # check values
    print(one_hot_encoding)
    print(correct_encoding)
    assert torch.all(one_hot_encoding == correct_encoding)


def test_operation_as_python():
    check_operation(metatensor.torch.one_hot)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(metatensor.torch.one_hot))


def test_save():
    scripted = torch.jit.script(metatensor.torch.one_hot)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
