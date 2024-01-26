from typing import Dict, List, Optional

import torch

from ..documentation import Labels, TensorBlock


class System:
    """
    A System contains all the information about an atomistic system; and should be used
    as the input of metatensor atomistic models.
    """

    def __init__(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ):
        """
        You can create a :py:class:`System` with ``species``, ``positions`` and ``cell``
        tensors, or convert data from other libraries.

        .. admonition:: Converting data to metatensor `System`

            Some external packages provides ways to create :py:class:`System` using data
            from other libraries:

            - `rascaline`_ has the :py:func:`rascaline.torch.systems_to_torch()`
              function that can convert from ASE, chemfiles and PySCF.

            .. _rascaline: https://luthaf.fr/rascaline/latest/index.html

        :param species: 1D tensor of integer representing the particles identity. For
            atoms, this is typically their atomic numbers.

        :param positions: 2D tensor of shape (len(species), 3) containing the Cartesian
            positions of all particles in the system.

        :param cell: 2D tensor of shape (3, 3), describing the bounding box/unit cell of
            the system. Each row should be one of the bounding box vector; and columns
            should contain the x, y, and z components of these vectors (i.e. the cell
            should be given in row-major order). Systems are assumed to obey periodic
            boundary conditions, non-periodic systems should set the cell to 0.
        """

    def __len__(self) -> int: ...

    @property
    def species(self) -> torch.Tensor:
        """Tensor of 32-bit integers representing the particles identity"""

    @property
    def positions(self) -> torch.Tensor:
        """
        Tensor of floating point values containing the particles cartesian coordinates
        """

    @property
    def cell(self) -> torch.Tensor:
        """Tensor of floating point values containing bounding box/cell of the system"""

    @property
    def device(self) -> torch.device:
        """get the device of all the arrays stored inside this :py:class:`System`"""

    @property
    def dtype(self) -> torch.dtype:
        """
        get the dtype of all the arrays stored inside this :py:class:`System`

        .. warning::

            Due to limitations in TorchScript C++ extensions, the dtype is returned as
            an integer, which can not be compared with :py:class:`torch.dtype`
            instances. See :py:meth:`TensorBlock.dtype` for more information.
        """

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> "System":
        """
        Move all the arrays in this system to the given ``dtype`` and ``device``.

        :param dtype: new dtype to use for all arrays. The dtype stays the same if this
            is set to ``None``.
        :param device: new device to use for all arrays. The device stays the same if
            this is set to ``None``.
        """

    def add_neighbors_list(
        self,
        options: "NeighborsListOptions",
        neighbors: TensorBlock,
    ):
        """
        Add a new neighbors list in this system corresponding to the given ``options``.

        The neighbors list should have the following samples: ``"first_atom"``,
        ``"second_atom"``, ``"cell_shift_a"``, ``"cell_shift_b"``, ``"cell_shift_c"``,
        containing the index of the first and second atoms (matching the "atom" sample
        in the positions); and the number of cell vector a/b/c to add to the positions
        difference to get the pair vector.

        The neighbors should also have a single component ``"xyz"`` with values ``[0, 1,
        2]``; and a single property ``"distance"`` with value 0.

        The neighbors values must contain the distance vector from the first to the
        second atom, i.e. ``positions[second_atom] - positions[first_atom] +
        cell_shift_a * cell_a + cell_shift_b * cell_b + cell_shift_c * cell_c``.

        :param options: options of the neighbors list
        :param neighbors: list of neighbors stored in a :py:class:`TensorBlock`
        """

    def get_neighbors_list(
        self,
        options: "NeighborsListOptions",
    ) -> TensorBlock:
        """
        Retrieve a previously stored neighbors list with the given ``options``, or throw
        an error if no such neighbors list exists.

        :param options: options of the neighbors list to retrieve
        """

    def known_neighbors_lists(self) -> List["NeighborsListOptions"]:
        """
        Get all the neighbors lists options registered with this :py:class:`System`
        """

    def add_data(self, name: str, data: TensorBlock):
        """
        Add custom data to this system, stored as :py:class:`TensorBlock`.

        This is intended for experimentation with models that need more data as input,
        and moved into a field of ``System`` later.

        :param name: name of the custom data
        :param data: values of the custom data
        """

    def get_data(self, name: str) -> TensorBlock:
        """
        Retrieve custom data stored in this System with the given ``name``, or throw
        an error if no data can be found.

        :param name: name of the custom data to retrieve
        """

    def known_data(self) -> List[str]:
        """
        Get the name of all the custom data registered with this :py:class:`System`
        """


class NeighborsListOptions:
    """Options for the calculation of a neighbors list"""

    def __init__(self, model_cutoff: float, full_list: bool, requestor: str = ""):
        """
        :param model_cutoff: spherical cutoff radius for the neighbors list, in the
            model units
        :param full_list: should the list be a full or half neighbors list
        :param requestor: who requested this neighbors list, you can add additional
            requestors later using :py:meth:`add_requestor`
        """

    @property
    def model_cutoff(self) -> float:
        """Spherical cutoff radius for this neighbors list in model units"""

    @property
    def engine_cutoff(self) -> float:
        """
        Spherical cutoff radius for this neighbors list in engine units. This defaults
        to the same value as ``model_cutoff`` until :py:meth:`set_engine_unit()` is
        called.
        """

    def set_engine_unit(self, conversion):
        """Set the conversion factor from the model units to the engine units"""

    @property
    def full_list(self) -> bool:
        """
        Should the list be a full neighbors list (contains both the pair ``i->j`` and
        ``j->i``) or a half neighbors list (contains only the pair ``i->j``)
        """

    def requestors(self) -> List[str]:
        """Get the list of modules requesting this neighbors list"""

    def add_requestor(self, requestor: str):
        """
        Add another ``requestor`` to the list of modules requesting this neighbors list
        """

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    def __eq__(self, other: "NeighborsListOptions") -> bool: ...

    def __ne__(self, other: "NeighborsListOptions") -> bool: ...


class ModelOutput:
    """Description of one of the quantity a model can compute."""

    def __init__(
        self,
        quantity: str = "",
        unit: str = "",
        per_atom: bool = False,
        explicit_gradients: List[str] = [],  # noqa B006
    ): ...

    quantity: str
    """
    Quantity of the output (e.g. energy, dipole, …).  If this is an empty
    string, no unit conversion will be performed.
    """

    unit: str
    """
    Unit of the output. If this is an empty string, no unit conversion will
    be performed.
    """

    per_atom: bool
    """Is the output defined per-atom or for the overall structure"""

    explicit_gradients: List[str]
    """
    Which gradients should be computed eagerly and stored inside the output
    :py:class:`TensorMap`.
    """


class ModelCapabilities:
    """Description of a model capabilities, i.e. everything a model can do."""

    def __init__(
        self,
        length_unit: str = "",
        species: List[int] = [],  # noqa B006
        outputs: Dict[str, ModelOutput] = {},  # noqa B006
    ): ...

    length_unit: str
    """unit of lengths the model expects as input"""

    species: List[int]
    """which atomic species the model can handle"""

    outputs: Dict[str, ModelOutput]
    """
    All possible outputs from this model and corresponding settings.

    During a specific run, a model might be asked to only compute a subset of these
    outputs.
    """


class ModelEvaluationOptions:
    """
    Options requested by the simulation engine/evaluation code when doing a single model
    evaluation.
    """

    def __init__(
        self,
        length_unit: str = "",
        outputs: Dict[str, ModelOutput] = {},  # noqa B006
        selected_atoms: Optional[Labels] = None,
    ): ...

    length_unit: str
    """unit of lengths the engine uses for the model input"""

    outputs: Dict[str, ModelOutput]
    """requested outputs for this run and corresponding settings"""

    selected_atoms: Optional[Labels]
    """
    Only run the calculation for a selected subset of atoms.

    If this is set to ``None``, run the calculation on all atoms. If this is a set of
    :py:class:`metatensor.torch.Labels`, it will have two dimensions named ``"system"``
    and ``"atom"``, containing the 0-based indices of all the atoms in the selected
    subset.
    """


def check_atomistic_model(path: str):
    """
    Check that the file at ``path`` contains an exported metatensor atomistic model, and
    that this model can be loaded in the current process.

    This function should be called before :py:func:`torch.jit.load()` when loading an
    existing model.
    """


def register_autograd_neighbors(
    system: System, neighbors: TensorBlock, check_consistency: bool
):
    """
    Register a new autograd node going from (``system.positions``, ``system.cell``) to
    the ``neighbors`` distance vectors.

    This does not recompute the distance vectors, but work as-if all the data in
    ``neighbors.values`` was computed directly from ``system.positions`` and
    ``system.cell``, allowing downstream models to use it directly with full autograd
    integration.

    :param system: system containing the positions and cell used to compute the
        neighbors list
    :param system: neighbors list, following the same format as
        :py:meth:`System.add_neighbors_list`
    :param check_consistency: can be set to ``True`` to run a handful of additional
        checks in case the data in neighbors does not follow what's expected.
    """
