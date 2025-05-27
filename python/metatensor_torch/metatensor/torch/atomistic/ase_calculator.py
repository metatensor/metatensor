import warnings


def __getattr__(name):
    warnings.warn(
        "`metatensor.torch.atomistic.ase_calculator` now lives in "
        "`metatomic.torch.ase_calculator`, please update your code",
        stacklevel=2,
    )
    raise AttributeError(name)
