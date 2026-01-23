import warnings


warnings.filterwarnings(
    "ignore", message=".*torch.jit.script.*is deprecated", category=DeprecationWarning
)


def __getattr__(name):
    warnings.warn(
        "`metatensor.torch.atomistic` now lives in "
        "`metatomic.torch`, please update your code",
        stacklevel=2,
    )
    raise AttributeError(name)
