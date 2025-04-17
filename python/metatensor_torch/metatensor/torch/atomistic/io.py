import warnings


def __getattr__(name):
    from metatomic.torch.io import load_system, save

    warnings.warn(
        "importing from metatensor.torch.atomistic.io is deprecated and "
        "will be removed in the next release.\nPlease add a depencency on "
        f"`metatomic[torch]` and import {name!r} from metatomic.torch.io instead.",
        stacklevel=2,
    )

    if name == "load_system":
        return load_system
    elif name == "save":
        return save

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
