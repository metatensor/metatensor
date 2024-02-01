from typing import Dict

import torch


@torch.jit.script
class Quantity:
    """
    A physical quantity that can be computed by a metatensor atomistic model, containing
    mainly information about units and units conversion for this quantity.
    """

    def __init__(self, baseline: str, conversions: Dict[str, float]):
        self._baseline = baseline
        """
        baseline unit used for this quantity, all the conversion factors should be
        given relative to this unit
        """

        self._conversions = conversions
        """
        Dictionary of conversion factors to the baseline unit. Unit names (the keys of
        this dictionary) are compared in a case- and spaces-insensitive manner.
        """

        assert self._baseline in self._conversions
        assert self._conversions[self._baseline] == 1.0

    def check_unit(self, unit: str) -> str:
        if unit != "":
            unit = unit.lower()
            unit = unit.replace(" ", "")  # remove spaces in e.g. `kcal /   mol`.
            if unit not in self._conversions:
                raise ValueError(
                    f"unknown unit '{unit}', only {list(self._conversions.keys())} "
                    "are supported"
                )
        return unit

    def conversion(self, from_unit: str, to_unit: str) -> float:
        """Get the conversion factor from ``from_unit`` to ``to_unit``"""
        from_unit = self.check_unit(from_unit)
        to_unit = self.check_unit(to_unit)

        if from_unit == "" or to_unit == "":
            return 1.0

        return self._conversions[to_unit] / self._conversions[from_unit]


KNOWN_QUANTITIES = {
    "length": Quantity(
        baseline="angstrom",
        conversions={
            "angstrom": 1.0,
            "bohr": 1.8897261258369282,
            "nanometer": 0.1,
            "nm": 0.1,
        },
    ),
    "energy": Quantity(
        baseline="ev",
        conversions={
            "ev": 1.0,
            "mev": 1000.0,
            "hartree": 0.03674932247495664,
            "kcal/mol": 23.060548012069496,
            "kj/mol": 96.48533288249877,
        },
    ),
}
