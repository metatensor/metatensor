from typing import Dict, List, Optional

import torch

from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, ModelRunOptions


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

    def get_forward_gradients(self) -> List[str]:
        return self._c.forward_gradients

    def set_forward_gradients(self, forward_gradients: List[str]):
        self._c.forward_gradients = forward_gradients


def test_output():
    class TestModule(torch.nn.Module):
        def forward(self, x: ModelOutputWrap) -> ModelOutputWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)


class ModelCapabilitiesWrap:
    def __init__(self):
        self._c = ModelCapabilities()

    def get_length_unit(self) -> str:
        return self._c.length_unit

    def set_length_unit(self, unit: str):
        self._c.length_unit = unit

    def get_species(self) -> List[int]:
        return self._c.species

    def set_species(self, species: List[int]):
        self._c.species = species

    def get_outputs(self) -> Dict[str, ModelOutput]:
        return self._c.outputs

    def get_output(self, name: str) -> ModelOutput:
        return self._c.outputs[name]

    def set_output(self, name: str, output: ModelOutput):
        self._c.outputs[name] = output


def test_capabilities():
    class TestModule(torch.nn.Module):
        def forward(self, x: ModelCapabilitiesWrap) -> ModelCapabilitiesWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)


class ModelRunOptionsWrap:
    def __init__(self):
        self._c = ModelRunOptions()

    def get_length_unit(self) -> str:
        return self._c.length_unit

    def set_length_unit(self, unit: str):
        self._c.length_unit = unit

    def get_selected_atoms(self) -> Optional[List[int]]:
        return self._c.selected_atoms

    def set_selected_atoms(self, selected_atoms: Optional[List[int]]):
        self._c.selected_atoms = selected_atoms

    def get_outputs(self) -> Dict[str, ModelOutput]:
        return self._c.outputs

    def get_output(self, name: str) -> ModelOutput:
        return self._c.outputs[name]

    def set_output(self, name: str, output: ModelOutput):
        self._c.outputs[name] = output


def test_run_options():
    class TestModule(torch.nn.Module):
        def forward(self, x: ModelRunOptionsWrap) -> ModelRunOptionsWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)
