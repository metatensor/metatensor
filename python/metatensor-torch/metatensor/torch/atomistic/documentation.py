from typing import Dict, List, Optional

from ..documentation import TensorBlock


class System:
    """
    A System contains all the information about an atomistic system; and should
    be used as the input of metatensor atomistic models.
    """

    positions: TensorBlock
    """
    Positions and types/species of of the atoms in the system.

    This block must have two samples names ``"atom"`` and ``"species"``, where
    ``"atom"`` is the index of the atom in the system; and ``"species"`` is the atomic
    species (typically --- but not limited to --- the atomic number).

    The block must have a single component ``"xyz"`` with values ``[0, 1, 2]``; and a
    single property ``"position"`` with value 0.

    The :py:class:`TensorBlock` values must contain the cartesian coordinates of the
    atoms in the system that the model should know about (typically all atoms, but this
    can be a subset of atoms, e.g. when using domain decomposition).
    """

    cell: TensorBlock
    """
    Unit cell/bounding box of the system. Non-periodic system should set all the values
    to 0.

    This block must have a single sample ``"_"`` with value 0; two components
    ``"cell_abc"`` and ``"xyz"`` both with values ``[0, 1, 2]``; and a single property
    ``"cell"`` with value 0. The values of the :py:class:`TensorBlock` then correspond
    to to the matrix of the cell vectors, in row-major order.
    """

    def __init__(self, positions: TensorBlock, cell: TensorBlock):
        ...

    def __len__(self) -> int:
        ...

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

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, other: "NeighborsListOptions") -> bool:
        ...

    def __ne__(self, other: "NeighborsListOptions") -> bool:
        ...


class ModelOutput:
    """Description of one of the quantity a model can compute."""

    def __init__(
        self,
        quantity: str = "",
        unit: str = "",
        per_atom: bool = False,
        forward_gradients: List[str] = [],  # noqa B006
    ):
        ...

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

    forward_gradients: List[str]
    """Which gradients should be computed in forward mode"""


class ModelCapabilities:
    """Description of a model capabilities, i.e. everything a model can do."""

    def __init__(
        self,
        length_unit: str = "",
        species: List[int] = [],  # noqa B006
        outputs: Dict[str, ModelOutput] = {},  # noqa B006
    ):
        ...

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


class ModelRunOptions:
    """Options requested by the simulation engine when running with a model"""

    def __init__(
        self,
        length_unit: str = "",
        selected_atoms: Optional[List[int]] = None,
        outputs: Dict[str, ModelOutput] = {},  # noqa B006
    ):
        ...

    length_unit: str
    """unit of lengths the engine uses for the model input"""

    selected_atoms: Optional[List[int]]
    """
    Only run the calculation for a selected subset of atoms. If this is set to ``None``,
    run the calculation on all atoms.
    """

    outputs: Dict[str, ModelOutput]
    """requested outputs for this run and corresponding settings"""
