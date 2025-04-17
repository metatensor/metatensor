import warnings


def __getattr__(name):
    from metatomic.torch.ase_calculator import (  # noqa: F401
        MetatomicCalculator,
        _compute_ase_neighbors,
        _full_3x3_to_voigt_6_stress,
    )

    warnings.warn(
        "importing from metatensor.torch.atomistic.ase_calculator is deprecated and "
        "will be removed in the next release.\nPlease add a depencency on "
        f"`metatomic[torch]` and import {name!r} from metatomic.torch.ase_calculator"
        " instead.",
        stacklevel=2,
    )

    if name == "MetatensorCalculator":
        warnings.warn(
            "'MetatensorCalculator' was renamed to 'MetatomicCalculator', "
            "please update your imports.",
            stacklevel=2,
        )
        return MetatomicCalculator

    if name in locals():
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
