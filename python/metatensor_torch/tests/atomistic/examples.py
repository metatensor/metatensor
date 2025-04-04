import importlib.util
import os
import sys

import torch

from metatensor.torch.atomistic import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
)


EXAMPLES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples")
)


def test_export_atomistic_model(tmp_path):
    """test if model defined in `1-export-atomistic-model.py` can run"""
    os.chdir(tmp_path)

    # import example from full path
    spec = importlib.util.spec_from_file_location(
        "export_atomistic_model",
        os.path.join(EXAMPLES, "atomistic", "1-export-atomistic-model.py"),
    )

    export_atomistic_model = importlib.util.module_from_spec(spec)
    sys.modules["export_atomistic_model"] = export_atomistic_model
    spec.loader.exec_module(export_atomistic_model)

    # define properties for prediction
    system = System(
        types=torch.tensor([1]),
        positions=torch.tensor([[1.0, 1, 1]], dtype=torch.float64, requires_grad=True),
        cell=torch.zeros([3, 3], dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )

    outputs = {
        "energy": ModelOutput(quantity="energy", unit="eV", sample_kind=["system"]),
    }

    # run bare model
    export_atomistic_model.model([system], outputs)

    # run exported model
    options = ModelEvaluationOptions(length_unit="Angstrom", outputs=outputs)
    export_atomistic_model.wrapper([system], options, check_consistency=True)

    # run exported and saved model
    export_atomistic_model.wrapper.save("exported-model.pt")
    atomistic_model = load_atomistic_model("exported-model.pt")
    atomistic_model([system], options, check_consistency=True)
