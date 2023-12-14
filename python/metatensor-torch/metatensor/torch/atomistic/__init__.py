import torch
import os

if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX") is not None:
    from .documentation import System, NeighborsListOptions
    from .documentation import ModelOutput, ModelEvaluationOptions, ModelCapabilities

    from .documentation import check_atomistic_model, register_autograd_neighbors

else:
    System = torch.classes.metatensor.System
    NeighborsListOptions = torch.classes.metatensor.NeighborsListOptions

    ModelOutput = torch.classes.metatensor.ModelOutput
    ModelEvaluationOptions = torch.classes.metatensor.ModelEvaluationOptions
    ModelCapabilities = torch.classes.metatensor.ModelCapabilities

    check_atomistic_model = torch.ops.metatensor.check_atomistic_model
    register_autograd_neighbors = torch.ops.metatensor.register_autograd_neighbors

from .model import MetatensorAtomisticModel  # noqa
