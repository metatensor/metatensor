import torch
import os

if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX") is not None:
    from .documentation import System, NeighborsListOptions
    from .documentation import ModelOutput, ModelRunOptions, ModelCapabilities
else:
    System = torch.classes.metatensor.System
    NeighborsListOptions = torch.classes.metatensor.NeighborsListOptions

    ModelOutput = torch.classes.metatensor.ModelOutput
    ModelRunOptions = torch.classes.metatensor.ModelRunOptions
    ModelCapabilities = torch.classes.metatensor.ModelCapabilities
