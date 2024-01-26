import datetime
import hashlib
import json
import os
import platform
import shutil
import site
import warnings
from typing import Dict, List, Optional

import torch

from .. import Labels, TensorBlock, TensorMap
from .. import __version__ as metatensor_version
from . import (
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelOutput,
    NeighborsListOptions,
    System,
)
from .units import KNOWN_QUANTITIES, Quantity


class MetatensorAtomisticModel(torch.nn.Module):
    """
    :py:class:`MetatensorAtomisticModel` is the main entry point for atomistic machine
    learning based on metatensor. It is the interface between custom, user-defined
    models and simulation engines. Users should wrap their models with this class, and
    use :py:meth:`export()` to save and export the model to a file. The exported models
    can then be loaded by a simulation engine to compute properties of atomistic
    systems.

    When wrapping a ``module``, you should declare what the model is capable of (using
    :py:class:`ModelCapabilities`). This includes what units the model expects as input
    and what properties the model can compute (using :py:class:`ModelOutput`). The
    simulation engine will then ask the model to compute some subset of these properties
    (through a :py:class:`metatensor.torch.atomistic.ModelEvaluationOptions`), on all or
    a subset of atoms of an atomistic system.

    Additionally, the wrapped ``module`` can request neighbors list to be computed by
    the simulation engine, and stored inside the input :py:class:`System`. This is done
    by defining ``requested_neighbors_lists(self) -> List[NeighborsListOptions]`` on the
    wrapped model or any of it's sub-module. :py:class:`MetatensorAtomisticModel` will
    unify identical requests before storing them and exposing it's own
    :py:meth:`requested_neighbors_lists()` that should be used by the engine to know
    what it needs to compute.

    There are several requirements on the wrapped ``module`` must satisfy. The main one
    is concerns the ``forward()`` function, which must have the following signature:


    >>> import torch
    >>> from typing import List, Dict, Optional
    >>> from metatensor.torch import Labels, TensorBlock
    >>> from metatensor.torch.atomistic import ModelOutput, System
    >>> class CustomModel(torch.nn.Module):
    ...     def forward(
    ...         self,
    ...         systems: List[System],
    ...         outputs: Dict[str, ModelOutput],
    ...         selected_atoms: Optional[Labels] = None,
    ...     ) -> Dict[str, TensorMap]: ...
    ...

    The returned dictionary should have the same keys as ``outputs``, and the values
    should contains the corresponding properties of the ``systems``, as computed for the
    subset of atoms defined in ``selected_atoms``. For some specific outputs, there are
    additional constrains on how the associated metadata should look like, documented in
    the :ref:`atomistic-models-outputs` section.

    Additionally, the wrapped ``module`` should not already be compiled by TorchScript,
    and should be in "eval" mode (i.e. ``module.training`` should be ``False``).

    For example, a custom module predicting the energy as a constant time the number of
    atoms could look like this

    >>> class ConstantEnergy(torch.nn.Module):
    ...     def __init__(self, constant: float):
    ...         super().__init__()
    ...         self.constant = torch.tensor(constant).reshape(1, 1)
    ...
    ...     def forward(
    ...         self,
    ...         systems: List[System],
    ...         outputs: Dict[str, ModelOutput],
    ...         selected_atoms: Optional[Labels] = None,
    ...     ) -> Dict[str, TensorMap]:
    ...         results: Dict[str, TensorMap] = {}
    ...         if "energy" in outputs:
    ...             if outputs["energy"].per_atom:
    ...                 raise NotImplementedError("per atom energy is not implemented")
    ...
    ...             dtype = systems[0].positions.dtype
    ...             energies = torch.zeros(len(systems), 1, dtype=dtype)
    ...             for i, system in enumerate(systems):
    ...                 if selected_atoms is None:
    ...                     n_atoms = len(system)
    ...                 else:
    ...                     n_atoms = len(selected_atoms)
    ...
    ...                 energies[i] = self.constant * n_atoms
    ...
    ...             systems_idx = torch.tensor([[i] for i in range(len(systems))])
    ...             energy_block = TensorBlock(
    ...                 values=energies,
    ...                 samples=Labels(["system"], systems_idx.to(torch.int32)),
    ...                 components=[],
    ...                 properties=Labels(["energy"], torch.IntTensor([[0]])),
    ...             )
    ...
    ...             results["energy"] = TensorMap(
    ...                 keys=Labels(["_"], torch.IntTensor([[0]])),
    ...                 blocks=[energy_block],
    ...             )
    ...
    ...         return results
    ...

    Wrapping and exporting this model would then look like this:

    >>> import os
    >>> import tempfile
    >>> from metatensor.torch.atomistic import MetatensorAtomisticModel
    >>> from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
    >>> model = ConstantEnergy(constant=3.141592)
    >>> # put the model in inference mode
    >>> model = model.eval()
    >>> # Define the model capabilities
    >>> capabilities = ModelCapabilities(
    ...     length_unit="angstrom",
    ...     species=[1, 2, 6, 8, 12],
    ...     outputs={
    ...         "energy": ModelOutput(
    ...             quantity="energy",
    ...             unit="eV",
    ...             per_atom=False,
    ...             explicit_gradients=[],
    ...         ),
    ...     },
    ... )
    >>> # wrap the model
    >>> wrapped = MetatensorAtomisticModel(model, capabilities)
    >>> # export the model
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     wrapped.export(os.path.join(directory, "constant-energy-model.pt"))
    ...
    """

    # Some annotation to make the TorchScript compiler happy
    _requested_neighbors_lists: List[NeighborsListOptions]
    _known_quantities: Dict[str, Quantity]

    def __init__(self, module: torch.nn.Module, capabilities: ModelCapabilities):
        """
        :param module: The torch module to wrap and export.
        :param capabilities: Description of the model capabilities.
        """
        super().__init__()

        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"`module` should be a torch.nn.Module, not {type(module)}")

        if isinstance(module, torch.jit.RecursiveScriptModule):
            raise TypeError("module should not already be a ScriptModule")

        if module.training:
            raise ValueError("module should not be in training mode")

        _check_annotation(module)
        self._module = module

        # ============================================================================ #

        # recursively explore `module` to get all the requested_neighbors_lists
        self._requested_neighbors_lists = []
        _get_requested_neighbors_lists(
            self._module,
            self._module.__class__.__name__,
            self._requested_neighbors_lists,
        )
        # ============================================================================ #

        self._capabilities = capabilities
        self._known_quantities = KNOWN_QUANTITIES

        length = self._known_quantities["length"]
        length.check_unit(self._capabilities.length_unit)

        # Check the units of the outputs
        for name, output in self._capabilities.outputs.items():
            if output.quantity == "":
                continue

            if output.quantity not in self._known_quantities:
                raise ValueError(
                    f"unknown output quantity '{output.quantity}' for '{name}' output, "
                    f"only {list(self._known_quantities.keys())} are supported"
                )

            quantity = self._known_quantities[output.quantity]
            quantity.check_unit(output.unit)

    def wrapped_module(self) -> torch.nn.Module:
        """Get the module wrapped in this :py:class:`MetatensorAtomisticModel`"""
        return self._module

    @torch.jit.export
    def capabilities(self) -> ModelCapabilities:
        """Get the capabilities of the wrapped model"""
        return self._capabilities

    @torch.jit.export
    def requested_neighbors_lists(
        self,
        length_unit: Optional[str] = None,
    ) -> List[NeighborsListOptions]:
        """
        Get the neighbors lists required by the wrapped model or any of the child
        module.

        :param length_unit: If not ``None``, this should contain a known unit of length.
            The returned neighbors lists will use this to set the ``engine_cutoff``
            field.
        """
        if length_unit is not None:
            length = self._known_quantities["length"]
            conversion = length.conversion(self._capabilities.length_unit, length_unit)
        else:
            conversion = 1.0

        for request in self._requested_neighbors_lists:
            request.set_engine_unit(conversion)

        return self._requested_neighbors_lists

    def forward(
        self,
        systems: List[System],
        options: ModelEvaluationOptions,
        check_consistency: bool,
    ) -> Dict[str, TensorMap]:
        """Run the wrapped model and return the corresponding outputs.

        Before running the model, this will convert the ``system`` data from the engine
        unit to the model unit, including all neighbors lists distances.

        After running the model, this will convert all the outputs from the model units
        to the engine units.

        :param system: input system on which we should run the model. The system should
            already contain all neighbors lists corresponding to the options in
            :py:meth:`requested_neighbors_lists()`.
        :param options: options for this run of the model
        :param check_consistency: Should we run additional check that everything is
            consistent? This should be set to ``True`` when verifying a model, and to
            ``False`` once you are sure everything is running fine.

        :return: A dictionary containing all the model outputs
        """

        if check_consistency:
            # check that the requested outputs match what the model can do
            _check_outputs(self._capabilities, options.outputs)

            # check that the species of the system match the one the model supports
            for system in systems:
                all_species = torch.unique(system.species)
                for species in all_species:
                    if species not in self._capabilities.species:
                        raise ValueError(
                            f"this model can not run for the atomic species '{species}'"
                        )

                # Check neighbors lists
                known_neighbors_lists = system.known_neighbors_lists()
                for request in self._requested_neighbors_lists:
                    found = False
                    for known in known_neighbors_lists:
                        if request == known:
                            found = True

                    if not found:
                        raise ValueError(
                            "missing neighbors list in the system: the model requested "
                            f"a list for {request}, but it was not computed and stored "
                            "in the system"
                        )

        # convert systems from engine to model units
        if self._capabilities.length_unit != options.length_unit:
            length = self._known_quantities["length"]
            conversion = length.conversion(
                from_unit=options.length_unit,
                to_unit=self._capabilities.length_unit,
            )

            systems = _convert_systems_units(
                systems,
                conversion,
                model_length_unit=self._capabilities.length_unit,
                system_length_unit=options.length_unit,
            )

        # run the actual calculations
        outputs = self._module(
            systems=systems,
            outputs=options.outputs,
            selected_atoms=options.selected_atoms,
        )

        # convert outputs from model to engine units
        for name, output in outputs.items():
            declared = self._capabilities.outputs[name]
            requested = options.outputs[name]
            if declared.quantity == "" or requested.quantity == "":
                continue

            if declared.quantity != requested.quantity:
                raise ValueError(
                    f"model produces values as '{declared.quantity}' for the '{name}' "
                    f"output, but the engine requested '{requested.quantity}'"
                )

            quantity = self._known_quantities[declared.quantity]
            conversion = quantity.conversion(
                from_unit=declared.unit, to_unit=requested.unit
            )

            if conversion != 1.0:
                for block in output.blocks():
                    block.values[:] *= conversion
                    for _, gradient in block.gradients():
                        gradient.values[:] *= conversion

        return outputs

    def export(self, file: str, collect_extensions: Optional[str] = None):
        """Export this model to a file that can then be loaded by simulation engine.

        :param file: where to save the model. This can be a path or a file-like object.
        :param collect_extensions: if not None, all currently loaded PyTorch extension
            will be collected in this directory. If this directory already exists, it
            is removed and re-created.
        """
        module = self.eval()
        try:
            module = torch.jit.script(module)
        except RuntimeError as e:
            raise RuntimeError("could not convert the module to TorchScript") from e

        # TODO: can we freeze these?
        # module = torch.jit.freeze(module)

        # record the list of loaded extensions, to check that they are also loaded when
        # executing the model.
        if collect_extensions is not None:
            if os.path.exists(collect_extensions):
                shutil.rmtree(collect_extensions)
            os.makedirs(collect_extensions)
            # TODO: the extensions are currently collected in a separate directory,
            # should we store the files directly inside the model file? This would makes
            # the model platform-specific but much more convenient (since the end user
            # does not have to move a model around)

        extensions = []
        for library in torch.ops.loaded_libraries:
            # Remove any site-package prefix
            path = library
            for site_packages in site.getsitepackages():
                if path.startswith(site_packages):
                    path = os.path.relpath(path, site_packages)
                    break

            if collect_extensions is not None:
                collect_path = os.path.join(collect_extensions, path)
                if os.path.exists(collect_path):
                    raise RuntimeError(
                        f"more than one extension would be collected at {collect_path}"
                    )

                os.makedirs(os.path.dirname(collect_path), exist_ok=True)
                shutil.copyfile(library, collect_path)

            # get the name of the library, excluding any shared object prefix/suffix
            name = os.path.basename(library)
            if name.startswith("lib"):
                name = name[3:]

            if name.endswith(".so"):
                name = name[:-3]

            if name.endswith(".dll"):
                name = name[:-4]

            if name.endswith(".dylib"):
                name = name[:-6]

            # Collect the hash of the extension shared library. We don't currently use
            # this, but it would allow for binary-level reproducibility later.
            with open(library, "rb") as fd:
                sha256 = hashlib.sha256(fd.read()).hexdigest()

            extensions.append({"path": path, "name": name, "sha256": sha256})

        # Metadata about where and when the model was exported
        export_metadata = {
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "platform": platform.machine() + "-" + platform.system(),
            # TODO: user/hostname?
        }

        if collect_extensions is not None:
            export_metadata["extensions_directory"] = str(collect_extensions)

        torch.jit.save(
            module,
            file,
            _extra_files={
                "torch-version": torch.__version__,
                "metatensor-version": metatensor_version,
                "extensions": json.dumps(extensions),
                "metadata": json.dumps(export_metadata),
            },
        )


def _get_requested_neighbors_lists(
    module: torch.nn.Module,
    name: str,
    requested: List[NeighborsListOptions],
):
    if hasattr(module, "requested_neighbors_lists"):
        for new_options in module.requested_neighbors_lists():
            new_options.add_requestor(name)

            already_requested = False
            for existing in requested:
                if existing == new_options:
                    already_requested = True
                    for requestor in new_options.requestors():
                        existing.add_requestor(requestor)

            if not already_requested:
                requested.append(new_options)

    for child_name, child in module.named_children():
        _get_requested_neighbors_lists(child, name + "." + child_name, requested)


def _check_annotation(module: torch.nn.Module):
    # check annotations on forward
    annotations = module.forward.__annotations__
    expected_arguments = [
        "systems",
        "outputs",
        "selected_atoms",
        "return",
    ]

    expected_signature = (
        "`forward(self, "
        "systems: List[System], "
        "outputs: Dict[str, ModelOutput], "
        "selected_atoms: Optional[Labels]"
        ") -> Dict[str, TensorMap]`"
    )

    if list(annotations.keys()) != expected_arguments:
        raise TypeError(
            "`module.forward()` takes unexpected arguments, expected signature is "
            + expected_signature
        )

    if annotations["systems"] != List[System]:
        raise TypeError(
            "`systems` argument must be a list of metatensor atomistic `System`, "
            f"not {annotations['system']}"
        )

    if annotations["outputs"] != Dict[str, ModelOutput]:
        raise TypeError(
            "`outputs` argument must be `Dict[str, ModelOutput]`, "
            f"not {annotations['outputs']}"
        )

    if annotations["selected_atoms"] != Optional[Labels]:
        raise TypeError(
            "`selected_atoms` argument must be `Optional[Labels]`, "
            f"not {annotations['selected_atoms']}"
        )

    if annotations["return"] != Dict[str, TensorMap]:
        raise TypeError(
            "`forward()` must return a `Dict[str, TensorMap]`, "
            f"not {annotations['return']}"
        )


def _check_outputs(capabilities: ModelCapabilities, outputs: Dict[str, ModelOutput]):
    for name, requested in outputs.items():
        if name not in capabilities.outputs:
            raise ValueError(
                f"this model can not compute '{name}', the implemented "
                f"outputs are {capabilities.outputs.keys()}"
            )

        possible = capabilities.outputs[name]

        for parameter in requested.explicit_gradients:
            if parameter not in possible.explicit_gradients:
                raise ValueError(
                    f"this model can not compute explicit gradients of '{name}' "
                    f"with respect to '{parameter}'"
                )

        if requested.per_atom and not possible.per_atom:
            raise ValueError(
                f"this model can not compute '{name}' per atom, only globally"
            )


def _convert_systems_units(
    systems: List[System],
    conversion: float,
    model_length_unit: str,
    system_length_unit: str,
) -> List[System]:
    if conversion == 1.0:
        return systems

    new_systems: List[System] = []
    for system in systems:
        new_system = System(
            species=system.species,
            positions=conversion * system.positions,
            cell=conversion * system.cell,
        )

        # also update the neighbors list distances
        for request in system.known_neighbors_lists():
            neighbors = system.get_neighbors_list(request)
            new_system.add_neighbors_list(
                request,
                TensorBlock(
                    values=conversion * neighbors.values,
                    samples=neighbors.samples,
                    components=neighbors.components,
                    properties=neighbors.properties,
                ),
            )

        known_data = system.known_data()
        if len(known_data) != 0:
            warnings.warn(
                "the model requires a different length unit "
                f"({model_length_unit}) than the system ({system_length_unit}), "
                f"but we don't know how to convert custom data ({known_data}) "
                "accordingly",
                stacklevel=2,
            )

        for data in known_data:
            new_system.add_data(data, system.get_data(data))

        new_systems.append(new_system)

    return new_systems
