import os

import torch


if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") != "0":
    from .documentation import (
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
    System = torch.classes.metatensor.System
    NeighborListOptions = torch.classes.metatensor.NeighborListOptions

    ModelOutput = torch.classes.metatensor.ModelOutput
    ModelEvaluationOptions = torch.classes.metatensor.ModelEvaluationOptions
    ModelCapabilities = torch.classes.metatensor.ModelCapabilities
    ModelMetadata = torch.classes.metatensor.ModelMetadata

    read_model_metadata = torch.ops.metatensor.read_model_metadata
    load_model_extensions = torch.ops.metatensor.load_model_extensions
    check_atomistic_model = torch.ops.metatensor.check_atomistic_model

    register_autograd_neighbors = torch.ops.metatensor.register_autograd_neighbors
    unit_conversion_factor = torch.ops.metatensor.unit_conversion_factor

from .io import load_system, save  # noqa: F401
from .model import (  # noqa: F401
    MetatensorAtomisticModel,
    ModelInterface,
    is_atomistic_model,
    load_atomistic_model,  # noqa: F401
)
from .systems_to_torch import systems_to_torch  # noqa: F401
