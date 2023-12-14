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
            "bohr": 0.5291772105638411,
            "nanometer": 10.0,
            "nm": 10.0,
        },
    ),
    "energy": Quantity(
        baseline="ev",
        conversions={
            "ev": 1.0,
            "mev": 1e-3,
            "hartree": 27.211386024367243,
            "kcal/mol": 0.04336410390059322,
            "kj/mol": 0.010364269574711572,
        },
    ),
}
