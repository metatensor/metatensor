from typing import Dict, List, Optional

import torch

from metatensor.torch import Labels
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelOutput,
)


class ModelOutputWrap:
    def __init__(self):
        self._c = ModelOutput()

    def get_unit(self) -> str:
        return self._c.unit

    def set_unit(self, unit: str):
        self._c.unit = unit

    def get_quantity(self) -> str:
        return self._c.quantity

    def set_quantity(self, quantity: str):
        self._c.quantity = quantity

    def get_per_atom(self) -> bool:
        return self._c.per_atom

    def set_per_atom(self, per_atom: bool):
        self._c.per_atom = per_atom

    def get_explicit_gradients(self) -> List[str]:
        return self._c.explicit_gradients

    def set_explicit_gradients(self, explicit_gradients: List[str]):
        self._c.explicit_gradients = explicit_gradients


def test_output():
    class TestModule(torch.nn.Module):
        def forward(self, x: ModelOutputWrap) -> ModelOutputWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)


class ModelCapabilitiesWrap:
    def __init__(self):
        self._c = ModelCapabilities()

    def get_outputs(self) -> Dict[str, ModelOutput]:
        return self._c.outputs

    def get_output(self, name: str) -> ModelOutput:
        return self._c.outputs[name]

    def set_output(self, name: str, output: ModelOutput):
        self._c.outputs[name] = output

    def get_atomic_types(self) -> List[int]:
        return self._c.atomic_types

    def set_atomic_types(self, atomic_types: List[int]):
        self._c.atomic_types = atomic_types

    def get_interaction_range(self) -> float:
        return self._c.interaction_range

    def set_interaction_range(self, interaction_range: float):
        self._c.interaction_range = interaction_range

    def get_length_unit(self) -> str:
        return self._c.length_unit

    def set_length_unit(self, unit: str):
        self._c.length_unit = unit

    def get_supported_devices(self) -> List[str]:
        return self._c.supported_devices

    def set_supported_devices(self, devices: List[str]):
        self._c.supported_devices = devices


def test_capabilities():
    class TestModule(torch.nn.Module):
        def forward(self, x: ModelCapabilitiesWrap) -> ModelCapabilitiesWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)


class ModelEvaluationOptionsWrap:
    def __init__(self):
        self._c = ModelEvaluationOptions()

    def get_length_unit(self) -> str:
        return self._c.length_unit

    def set_length_unit(self, unit: str):
        self._c.length_unit = unit

    def get_selected_atoms(self) -> Optional[Labels]:
        return self._c.selected_atoms

    def set_selected_atoms(self, selected_atoms: Optional[Labels]):
        self._c.selected_atoms = selected_atoms

    def get_outputs(self) -> Dict[str, ModelOutput]:
        return self._c.outputs

    def get_output(self, name: str) -> ModelOutput:
        return self._c.outputs[name]

    def set_output(self, name: str, output: ModelOutput):
        self._c.outputs[name] = output


def test_run_options():
    class TestModule(torch.nn.Module):
        def forward(self, x: ModelEvaluationOptionsWrap) -> ModelEvaluationOptionsWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)
