import os
import warnings


def __getattr__(name):
    if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") != "0":
        from metatomic.torch.documentation import (
            ModelCapabilities,
            ModelEvaluationOptions,
            ModelMetadata,
            ModelOutput,
            NeighborListOptions,
            System,
            check_atomistic_model,
            load_model_extensions,
            read_model_metadata,
            register_autograd_neighbors,
            unit_conversion_factor,
        )

    else:
        from metatomic.torch import (  # noqa: F401
            ModelCapabilities,
            ModelEvaluationOptions,
            ModelMetadata,
            ModelOutput,
            NeighborListOptions,
            System,
            check_atomistic_model,
            load_model_extensions,
            read_model_metadata,
            register_autograd_neighbors,
            unit_conversion_factor,
        )

    from metatomic.torch import (  # noqa: F401
        MetatensorAtomisticModel,
        ModelInterface,
        is_atomistic_model,
        load_atomistic_model,
        systems_to_torch,
    )
    from metatomic.torch.io import load_system, save  # noqa: F401

    warnings.warn(
        "importing from metatensor.torch.atomistic is deprecated and will be removed "
        "in the next release.\nPlease add a depencency on `metatomic[torch]` and "
        f"import {name!r} from metatomic.torch instead.",
        stacklevel=2,
    )

    if name in locals():
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
