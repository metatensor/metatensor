from typing import Dict, List, Optional

import torch

from .. import TensorBlock
from . import ModelCapabilities, ModelRunOptions, NeighborsListOptions, System
from .units import KNOWN_QUANTITIES, Quantity


class MetatensorAtomisticModule(torch.nn.Module):
    """
    :py:class:`MetatensorAtomisticModule` is the main entry point for atomistic machine
    learning based on metatensor. It is the interface between custom, user-defined
    models and simulation engines. Users should wrap their models with this class, and
    use :py:meth:`export()` to save and export the model to a file. The exported models
    can then be loaded by a simulation engine to compute properties of atomistic
    systems.

    When wrapping a ``module``, you should declare what the model is capable of (using
    :py:class:`ModelCapabilities`). This includes what units the model expects as input
    and what properties the model can compute (using :py:class:`ModelOutput`). The
    simulation engine will then ask the model to compute some subset of these properties
    (through a :py:class:`metatensor.torch.atomistic.ModelRunOptions`), on all or a
    subset of atoms of an atomistic system.

    Additionally, the wrapped ``module`` can request neighbors list to be computed by
    the simulation engine, and stored inside the input :py:class:`System`. This is done
    by defining ``requested_neighbors_lists(self) -> List[NeighborsListOptions]`` on the
    wrapped model or any of it's sub-module. :py:class:`MetatensorAtomisticModule` will
    unify identical requests before storing them and exposing it's own
    :py:meth:`requested_neighbors_lists()` that should be used by the engine to know
    what it needs to compute.

    There are several requirements on the wrapped ``module`` must satisfy. The main one
    is concerns the ``forward()`` function, which must have the following signature:


    >>> import torch
    >>> from metatensor.torch import Labels, TensorBlock
    >>> from metatensor.torch.atomistic import ModelRunOptions, System
    >>> class CustomModule(torch.nn.Module):
    ...     def forward(
    ...         self, system: System, run_option: ModelRunOptions
    ...     ) -> Dict[str, TensorBlock]:
    ...         ...
    ...

    The returned dictionary should have the same keys as ``run_option.outputs``, and the
    values should contains the corresponding properties of the ``system``, as computed
    for the subset of atoms defined in ``run_options.selected_atoms``.

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
    ...         self, system: System, run_options: ModelRunOptions
    ...     ) -> Dict[str, TensorBlock]:
    ...         outputs: Dict[str, TensorBlock] = {}
    ...         if "energy" in run_options.outputs:
    ...             if run_options.outputs["energy"].per_atom:
    ...                 raise NotImplementedError("per atom energy is not implemented")
    ...
    ...             selected_atoms = run_options.selected_atoms
    ...             if selected_atoms is None:
    ...                 n_atoms = len(system)
    ...             else:
    ...                 n_atoms = len(selected_atoms)
    ...
    ...             outputs["energy"] = TensorBlock(
    ...                 values=self.constant * n_atoms,
    ...                 samples=Labels(["_"], torch.IntTensor([[0]])),
    ...                 components=[],
    ...                 properties=Labels(["energy"], torch.IntTensor([[0]])),
    ...             )
    ...
    ...         return outputs
    ...

    Wrapping and exporting this model would then look like this:

    >>> import os
    >>> import tempfile
    >>> from metatensor.torch.atomistic import MetatensorAtomisticModule
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
    ...             forward_gradients=[],
    ...         ),
    ...     },
    ... )
    >>> # wrap the model
    >>> wrapped = MetatensorAtomisticModule(model, capabilities)
    >>> # export the model
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     wrapped.export(os.path.join(directory, "constant-energy-model.pt"))
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
        system: System,
        run_options: ModelRunOptions,
        check_consistency: bool,
    ) -> Dict[str, TensorBlock]:
        """Run the wrapped model and return the corresponding outputs.

        Before running the model, this will convert the ``system`` data from the engine
        unit to the model unit, including all neighbors lists distances.

        After running the model, this will convert all the outputs from the model units
        to the engine units.

        :param system: input system on which we should run the model. The system should
            already contain all neighbors lists corresponding to the options in
            :py:meth:`requested_neighbors_lists()`.
        :param run_options: _description_
        :param check_consistency: Should we run additional check that everything is
            consistent? This should be set to ``True`` when verifying a model, and to
            ``False`` once you are sure everything is running fine.

        :return: A dictionary containing all the model outputs
        """

        if check_consistency:
            # check that the requested outputs match what the model can do
            _check_outputs(self._capabilities, run_options)

            # check that the species of the system match the one the model supports
            all_species = torch.unique(system.positions.samples.column("species"))
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
                        "missing neighbors list in the system: the model requested a "
                        f"list for {request}, but it was not computed and stored in "
                        "the system"
                    )

        # convert systems from engine to model units
        if self._capabilities.length_unit != run_options.length_unit:
            length = self._known_quantities["length"]
            conversion = length.conversion(
                from_unit=run_options.length_unit,
                to_unit=self._capabilities.length_unit,
            )

            system.positions.values[:] *= conversion
            system.cell.values[:] *= conversion

            # also update the neighbors list distances
            for options in self._requested_neighbors_lists:
                neighbors = system.get_neighbors_list(options)
                neighbors.values[:] *= conversion

        # run the actual calculations
        outputs = self._module(system=system, run_options=run_options)

        # convert outputs from model to engine units
        for name, output in outputs.items():
            declared = self._capabilities.outputs[name]
            requested = run_options.outputs[name]
            if declared.quantity == "" or requested.quantity == "":
                continue

            if declared.quantity != requested.quantity:
                raise ValueError(
                    f"model produces values as '{declared.quantity}' for the '{name}' "
                    f"output, but the engine requested '{requested.quantity}'"
                )

            quantity = self._known_quantities[declared.quantity]
            output.values[:] *= quantity.conversion(
                from_unit=declared.unit, to_unit=requested.unit
            )

        return outputs

    def export(self, file):
        """Export this model to a file that can then be loaded by simulation engine.

        :param file: where to save the model. This can be a path or a file-like object.
        """

        module = self.eval()
        try:
            module = torch.jit.script(module)
        except RuntimeError as e:
            raise RuntimeError("could not convert the module to TorchScript") from e

        # TODO: can we freeze these?
        # module = torch.jit.freeze(module)

        # TODO: record torch version

        # TODO: record list of loaded extensions

        torch.jit.save(module, file, _extra_files={})


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
        "system",
        "run_options",
        "return",
    ]

    expected_signature = (
        "`forward(self, system: System, run_options: ModelRunOptions) -> "
        "Dict[str, TensorBlock]`"
    )

    if list(annotations.keys()) != expected_arguments:
        raise TypeError(
            "`module.forward()` takes unexpected arguments, expected signature is "
            + expected_signature
        )

    if annotations["system"] != System:
        raise TypeError(
            "`system` argument must be a metatensor atomistic `System`, "
            f"not {annotations['system']}"
        )

    if annotations["run_options"] != ModelRunOptions:
        raise TypeError(
            "`run_options` argument must be a metatensor atomistic `ModelRunOptions`, "
            f"not {annotations['run_options']}"
        )

    if annotations["return"] != Dict[str, TensorBlock]:
        raise TypeError(
            "`forward()` must return a `Dict[str, TensorBlock]`, "
            f"not {annotations['return']}"
        )


def _check_outputs(capabilities: ModelCapabilities, run_options: ModelRunOptions):
    for name, requested in run_options.outputs.items():
        if name not in capabilities.outputs:
            raise ValueError(
                f"this model can not compute '{name}', the implemented "
                f"outputs are {capabilities.outputs.keys()}"
            )

        possible = capabilities.outputs[name]

        for parameter in requested.forward_gradients:
            if parameter not in possible.forward_gradients:
                raise ValueError(
                    f"this model can not compute gradients of '{name}' with respect to "
                    f"'{parameter}' in forward mode"
                )

        if requested.per_atom and not possible.per_atom:
            raise ValueError(
                f"this model can not compute '{name}' per atom, only globally"
            )
