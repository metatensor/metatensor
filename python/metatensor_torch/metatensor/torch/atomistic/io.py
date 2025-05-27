import warnings


def __getattr__(name):
    warnings.warn(
        "`metatensor.torch.atomistic.io` now lives in "
        "`metatomic.torch.io`, please update your code",
        stacklevel=2,
    )
    raise AttributeError(name)
