import warnings


def __getattr__(name):
    warnings.warn(
        "`metatensor.torch.atomistic` now lives in "
        "`metatomic.torch`, please update your code",
        stacklevel=2,
    )
    raise AttributeError(name)
